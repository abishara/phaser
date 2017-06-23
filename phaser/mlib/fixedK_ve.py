import os
import numpy as np
import math
import random
from scipy.misc import logsumexp
from collections import defaultdict, Counter
import h5py
import pandas
import pysam

import util
import hyper_params as hp

num_iterations = 12

def score(A, H):
  M_, N_ = A.shape
  K, _ = H.shape
  logP = 0.0
  M  = np.zeros((M_, K))
  MM = np.zeros((M_, K))
  S  = np.zeros((M_, K))
  for i in xrange(M_):
    S_i = np.zeros(K)
    for k in xrange(K):
      M[i,k] = sum(np.abs(A[i,np.sign(A[i,:]) == H[k,:]]))
      MM[i,k] = sum(np.abs(A[i,np.sign(A[i,:]) == -H[k,:]]))
      S[i,k] = (hp.prob_const["logP(h)"] +
        hp.prob_const["logP(err')"] * M[i,k] + 
        hp.prob_const["logP(err)"] * MM[i,k])
    logP += logsumexp(S[i,:])
  return logP, M, MM, S

def phase(scratch_path):
  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A = util.load_phase_inputs(h5_path)

  print 'loaded {} X {}'.format(len(bcodes), len(snps))
  bcodes, A = util.subsample_reads(bcodes, A, lim=5000)
  print '  - subsample to {} X {}'.format(len(bcodes), len(snps))

  M_, N_ = A.shape

  # hidden haplotypes
  H, _ = util.get_initial_state(A, hp.K)

  # initialize and save intermediate values for fast vectorized
  # computation
  logP, M, MM, S = score(A, H)
  logP_min = logP
  H_min = np.array(H)
  print 'initial logP', logP

  sidx = 0
  H_samples = np.zeros((num_iterations, hp.K, N_))
  for iteration in xrange(num_iterations):
    H_samples[sidx,:,:] = H
    sidx = (sidx + 1) % num_iterations
    print 'iteration', iteration
    for k_p in xrange(hp.K):
      print '  - hap {}, logP: {}'.format(k_p, logP)
      for j_p in xrange(N_):
        
        # skip any entries for which no barcodes overlap this genotype position
        if not (A[:,j_p] != 0).any():
          continue

        # toggle matches/mismatches at locus j for hap k
        pp = H[k_p, j_p]
        sel_rids = np.nonzero(A[:,j_p])[0]

        m = np.abs((pp * A[sel_rids, j_p]).clip(min=0))
        mm = np.abs((pp * A[sel_rids,j_p]).clip(max=0))
        M_k  =  M[sel_rids, k_p]
        MM_k = MM[sel_rids, k_p]
        M_k_p  =  M_k - m + mm
        MM_k_p = MM_k + m - mm
        # determine updated read scores for hap k
        S_k_p = (
          hp.prob_const["logP(h)"] +
          hp.prob_const["logP(err')"] * M_k_p + 
          hp.prob_const["logP(err)"] * MM_k_p
        )
        # compute base logP
        logP_p_vec = np.apply_along_axis(
          logsumexp,
          1,
          S[sel_rids,:],
        )
        # compute updated logP using updated read scores for hap k
        sel_ks = np.ones(hp.K, dtype=bool)
        sel_ks[k_p] = False
        logP_n_vec = np.apply_along_axis(
          logsumexp,
          1,
          np.hstack([S_k_p[:,None], S[np.ix_(sel_rids,sel_ks)]]),
        )
        delta_logP = sum(logP_n_vec) - sum(logP_p_vec)
        P = min(1, np.exp(delta_logP))
        #print 'delta_logP', delta_logP
        #die

        if P > random.random():
          #print '  - accept', delta_logP
          logP += delta_logP
          # toggle haps
          H[k_p,j_p] *= -1
          # update state
          M[sel_rids,k_p]  = M_k_p
          MM[sel_rids,k_p] = MM_k_p
          S[sel_rids,k_p] = S_k_p
        else:
          pass
          #print '  - reject'

        # save new minimum
        if logP > logP_min:
          logP_min = logP
          H_min = np.array(H)
          #print '**new minimum', logP
        #die
      #die

  #print 'converged', H_min
  h5_path = os.path.join(scratch_path, 'phased.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('H', data=H_min)
  h5f.create_dataset('H_samples', data=H_samples)
  h5f.close()

def make_outputs(inbam_path, scratch_path):
  inputsh5_path = os.path.join(scratch_path, 'inputs.h5')
  phaseh5_path = os.path.join(scratch_path, 'phased.h5')
  snps, bcodes, A = util.load_phase_inputs(inputsh5_path)
  h5f = h5py.File(phaseh5_path, 'r')
  H = np.array(h5f['H'])
  H_samples = np.array(h5f['H_samples'])
  h5f.close()

  print 'loaded {} X {}'.format(len(bcodes), len(snps))
  bcodes, A = util.subsample_reads(bcodes, A, lim=5000)
  print '  - subsample to {} X {}'.format(len(bcodes), len(snps))

  K, _ = H.shape
  M_, N_ = A.shape
  idx_rid_map = dict(list(enumerate(bcodes)))
  idx_snp_map = dict(list(enumerate(snps)))

  # determine final haplotypes from samples
  num_samples = num_iterations-4
  # convert ref from -1 to 0 so can compute sampled probability
  H_samples[H_samples == -1] = 0
  p_H = np.sum(H_samples[4:4+num_samples,:,:],axis=0) / num_samples
  K_, N_ = p_H.shape
  #H = np.zeros((K_, N_))
  #H[p_H >= 0.7] = 1
  #H[p_H < 0.03] = -1

  # determine read assignment to haplotypes
  W = np.empty((M_,))
  W.fill(-1)

  logP, M, MM, S = score(A, H)
  S_n = np.zeros((M_, K))
  for i in xrange(M_):
    # renormalize across K haps for this read
    S_n[i,:] = S[i,:] - logsumexp(S[i,:])
    # only assign if majority rule
    assn = np.argmax(S_n[i,:])
    if np.exp(S_n[i,assn]) > 0.8:
      W[i] = assn
    else:
      pass

  outdir_path = os.path.join(scratch_path, 'bins')
  util.mkdir_p(outdir_path)

  def get_bcodes_from_rids(rids):
    bcodes = set()
    for rid in np.nditer(rids):
      rid = int(rid)
      bcodes.add(idx_rid_map[rid])
    return bcodes
  
  clusters_map = {}
  for k in xrange(K):
    sel_rids = np.nonzero(W == k)[0]
    if len(sel_rids) > 0:
      bcode_set = get_bcodes_from_rids(sel_rids)
    else:
      bcode_set = set()
    clusters_map[k] = bcode_set

  util.make_bin_outputs(
    clusters_map,
    inbam_path,
    outdir_path,
  )

  h5_path = os.path.join(scratch_path, 'bins', 'clusters.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('H', data=H)
  h5f.create_dataset('W', data=W)
  h5f.close()

  print 'total haplotype positions', K_ * N_
  print '  - assigned', np.sum(np.abs(H))

  print 'assigned barcodes', sum((W != -1))
  print 'unassigned barcodes', sum((W == -1))

  return clusters_map


