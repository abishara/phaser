import os
import numpy as np
import math
import random
from scipy.misc import logsumexp
from collections import defaultdict, Counter
import h5py

import util
import debug

#K = 5
K = 10
prob_const = {
  "logP(h)" : math.log(1. / K),
  "logP(err)"  : math.log(0.01),
  "logP(err')" : math.log(1 - 0.01),
} 

def score(A, H):
  M_,N_ = A.shape
  logP = 0.0
  M  = np.zeros((M_, K))
  MM = np.zeros((M_, K))
  S  = np.zeros((M_, K))
  for i in xrange(M_):
    S_i = np.zeros(K)
    for k in xrange(K):
      M[i,k] = sum(np.abs(A[i,np.sign(A[i,:]) == H[k,:]]))
      MM[i,k] = sum(np.abs(A[i,np.sign(A[i,:]) == -H[k,:]]))
      S[i,k] = (prob_const["logP(h)"] +
        prob_const["logP(err')"] * M[i,k] + 
        prob_const["logP(err)"] * MM[i,k])
    logP += logsumexp(S[i,:])
  return logP, M, MM, S

def phase(scratch_path):
  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A = util.load_phase_inputs(h5_path)

  print 'loaded {} X {}'.format(len(bcodes), len(snps))

  M_, N_ = A.shape

  ## FIXME remove
  ## fix A to be 1, -1 and not allow barcodes to have multiple matches per
  ## site
  #print 'prev calls +1:{}, -1:{}'.format(np.sum(A[A>0]), np.sum(A[A<0]))
  #A[A > 0] = 1
  #A[A < 0] = -1
  #print 'post calls +1:{}, -1:{}'.format(np.sum(A[A>0]), np.sum(A[A<0]))

  # hidden haplotypes
  H = np.ones((K, N_))

  # initialize and save intermediate values for fast vectorized
  # computation
  logP, M, MM, S = score(A, H)
  logP_min = logP
  H_min = np.array(H)
  print 'initial logP', logP

  for iteration in xrange(12):
    print 'iteration', iteration
    for k_p in xrange(K):
      print '  - hap {}, logP: {}'.format(k_p, logP)
      for j_p in xrange(N_):

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
          prob_const["logP(h)"] +
          prob_const["logP(err')"] * M_k_p + 
          prob_const["logP(err)"] * MM_k_p
        )
        # compute base logP
        logP_p_vec = np.apply_along_axis(
          logsumexp,
          1,
          S[sel_rids,:],
        )
        # compute updated logP using updated read scores for hap k
        sel_ks = np.ones(K, dtype=bool)
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
  h5f.close()

def make_outputs(scratch_path):
  inputsh5_path = os.path.join(scratch_path, 'inputs.h5')
  phaseh5_path = os.path.join(scratch_path, 'phased.h5')
  snps, bcodes, A = util.load_phase_inputs(inputsh5_path)
  h5f = h5py.File(phaseh5_path, 'r')
  H = np.array(h5f['H'])
  h5f.close()
  M_, N_ = A.shape
  idx_rid_map = dict(list(enumerate(bcodes)))
  idx_snp_map = dict(list(enumerate(snps)))

  print 'loaded {} X {}'.format(len(bcodes), len(snps))

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
  debugdir_path = os.path.join(scratch_path, 'debug')
  util.mkdir_p(outdir_path)
  util.mkdir_p(debugdir_path)

  debug_assn_path = os.path.join(debugdir_path, 'assn.txt')

  s_idx = np.argsort(W)
  with open(debug_assn_path, 'w') as fout:
    for i in np.nditer(s_idx):
      i = int(i)
      fout.write('bcode:{}, assn:{}\n'.format(
        idx_rid_map[i], W[i],
      ))
      fout.write('  {}\n'.format(np.array_str(S[i,:],  max_line_width=200)))
      fout.write('  {}\n'.format(np.array_str(S_n[i,:],max_line_width=200)))

  clusters_map = {}
  for k in xrange(K):
    out_path = os.path.join(outdir_path, '{}.bin.txt'.format(k)) 
    sel_rids = np.nonzero(W == k)[0]
    bcode_set = set()
    with open(out_path, 'w') as fout:
      for rid in np.nditer(sel_rids):
        rid = int(rid)
        bcode = idx_rid_map[rid]
        fout.write('{}\n'.format(bcode))
        bcode_set.add(bcode)
    clusters_map[k] = bcode_set

    debug_path = os.path.join(debugdir_path, 'c.{}'.format(k))
    util.mkdir_p(debug_path)
    A_k = A[sel_rids,:]
    print 'debug for bin', k
    debug.dump(
      debug_path,
      H[k,:],
      A_k,
      sel_rids,
      idx_rid_map,
      idx_snp_map,
    )

  out_path = os.path.join(scratch_path, 'bins', 'clusters.p')
  util.write_pickle(out_path, clusters_map)
  print 'assigned barcodes', sum((W != -1))
  print 'unassigned barcodes', sum((W == -1))

