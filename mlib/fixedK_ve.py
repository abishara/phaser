import os
import numpy as np
import math
import random
from scipy.misc import logsumexp
from collections import defaultdict, Counter
import h5py

import util

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

  print 'loaded {} X {}'.format(len(bcodes), len(snps))

  missing_bcodes = set()
  #with open(os.path.join(scratch_path, 'missing-bcodes.txt')) as fin:
  #  for line in fin:
  #    bcode = line.strip()
  #    missing_bcodes.add(bcode)

  # determine read assignment to haplotypes
  W = np.empty((M_,))
  W.fill(-1)

  logP, M, MM, S = score(A, H)
  missing_cnt = Counter()
  for i in xrange(M_):
    bcode = idx_rid_map[i]
    # renormalize across K haps for this read
    S[i,:] -= logsumexp(S[i,:])
    # only assign if majority rule
    assn = np.argmax(S[i,:])
    if np.exp(S[i,assn]) > 0.8:
      W[i] = assn
      if bcode in missing_bcodes:
        missing_cnt[assn] += 1
        #if assn == 1:
        #  print '(m, mm):{},{}'.format(M[i,assn], MM[i,assn])
    else:
      if bcode in missing_bcodes:
        missing_cnt[-1] += 1

  print 'missing hits', missing_cnt.most_common()
  #die

  outdir_path = os.path.join(scratch_path, 'bins')
  util.mkdir_p(outdir_path)
  clusters_map = {}
  for k in xrange(K):
    out_path = os.path.join(outdir_path, '{}.bin.txt'.format(k)) 
    sel_rids = np.nonzero(W == k)[0]
    if True:
    #if k == 1:

      mA = A[sel_rids,:].copy() * H[k,:]
      mmA = A[sel_rids,:].copy() * H[k,:]
      mA[mA < 0] = 0
      mA = np.abs(mA)
      mmA[mmA > 0] = 0
      mmA = np.abs(mmA)
      sel_snps = np.any(A[sel_rids,:] != 0, axis=0)

      hapM = np.sum(mA,axis=0)
      hapMM = np.sum(mmA, axis=0)

      m = hapM[sel_snps]
      mm = hapMM[sel_snps]

      diffs = sorted(zip(mm, m),reverse=True)

      mmgt = 0
      mgt = 0
      for mm, m in diffs:
        if mm > m:
          mmgt += 1
        else:
          mgt += 1
      print 'k', k
      print '  more mm', mmgt
      print '  more m', mgt
      print '  diffs', diffs[:10]
      print '  end diffs', diffs[-10:]
      print


    bcode_set = set()
    with open(out_path, 'w') as fout:
      for rid in np.nditer(sel_rids):
        rid = int(rid)
        bcode = idx_rid_map[rid]
        fout.write('{}\n'.format(bcode))
        bcode_set.add(bcode)
    clusters_map[k] = bcode_set

  out_path = os.path.join(scratch_path, 'bins', 'clusters.p')
  write_pickle(out_path, clusters_map)
  print 'assigned barcodes', sum((W != -1))
  print 'unassigned barcodes', sum((W == -1))
