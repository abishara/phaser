import os
import sys
import numpy as np
import math
import random
from scipy.misc import logsumexp
from scipy.special import gammaln
from collections import defaultdict, Counter
import h5py

import util

(alpha, beta) = (5., 1.)
#(alpha, beta) = (1., 1.)
K = 5
#K = 10

# normalization constant for the beta
beta_R = (
  gammaln(alpha + beta)
  - gammaln(alpha) +
  - gammaln(beta)
)

def score_beta_site(m, mm):
  a = gammaln(m + alpha)
  b = gammaln(mm + beta)
  n = gammaln(m + mm + alpha + beta)
  return a, b, n

def score(A, H, C):
  M_, N_ = A.shape
  logP = 0.0
  # match+mismatch counts for each site in each hap K
  M  = np.zeros((K, N_))
  MM = np.zeros((K, N_))
  # gamma counts
  G = np.zeros((K, N_, 3))
  # cluster scores with current assignment
  S = np.zeros((K, 1))

  # score assigned reads in each cluser by integrating over thetas
  for k in xrange(K):
    for j in xrange(N_):
      C_i = (C == k)
      A_k = A[C_i,:]
      M[k,j]  = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[k, j], j]))
      MM[k,j] = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[k, j], j]))
      G[k,j,:] = score_beta_site(M[k,j], MM[k,j])
    S[k] = (
      N_ * beta_R + 
      np.sum(G[k,:,0]) + 
      np.sum(G[k,:,1]) -
      np.sum(G[k,:,2])
    )
  logP = np.sum(S)

  return logP, M, MM, G, S

def get_initial_state(A):
  M_, N_ = A.shape

  # get inspired by some reads to initialize the hidden haplotypes
  #FIXME TODO
  H = np.ones((K, N_))
  C = np.random.choice(K, N_)

  return H, C

def score_read(m, mm, M, MM):
  logP = 0.

  mask = ((m > 0) | (mm > 0))
  N = M + MM
  # match component
  logP += np.sum((np.log(alpha + M[mask]) - np.log(alpha + beta + N[mask] - 1)) * m[mask])
  # mismatch component
  logP += np.sum((np.log(alpha + MM[mask]) - np.log(alpha + beta + N[mask] - 1)) * mm[mask])

  return logP

def sample_haplotype(M, MM, G, H_p, A_k):
  N = M.shape[0]
  H_n = np.zeros(N)
  #H_n = np.zeros((1, N))
  G_n = np.zeros((N, 3))

  def assert_hap_state(H):
    for j in xrange(N):
      M_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[j], j]))
      MM_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[j], j]))
      assert M[j]  == M_c
      assert MM[j] == MM_c

  assert_hap_state(H_p)
  # get flipped gammas for all sites
  G_n[:,0], G_n[:,1], G_n[:,2] = score_beta_site(MM, M)
  logP_p = G[:,0] + G[:,1]
  logP_n = G_n[:,0] + G_n[:,1]
  # renormalize
  logP_flip = logP_n - np.apply_along_axis(
    logsumexp,
    0,
    np.vstack((logP_p, logP_n)),
  )
  #print 'H_p', H_p
  #print 'A_k', A_k
  #print 'flip prob', np.exp(logP_flip)
  #print
  # choose bits to flip
  flip_mask = np.random.random(N) < np.exp(logP_flip)
  # flip selected bits for newly sampled haplotype
  H_n = np.array(H_p)
  H_n[flip_mask] = -H_p[flip_mask]
  # update sufficient statistics for loci that flipped
  fm = flip_mask
  MM[fm], M[fm], G[:,0], G[:,1] = M[fm], MM[fm], G[:,1], G[:,0]
  assert_hap_state(H_n)

  return H_n

def phase(scratch_path):
  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A = util.load_phase_inputs(h5_path)

  print 'loaded {} X {}'.format(len(bcodes), len(snps))

  M_, N_ = A.shape

  # hidden haplotypes
  H, C  = get_initial_state(A)

  # initialize and save intermediate values for fast vectorized
  # computation
  logP, M, MM, G, S = score(A, H, C)
  logP_min = logP
  print 'initial logP', logP

  def get_hap_counts(r, H_k):
    m  = np.abs((H_k * r).clip(min=0))
    mm = np.abs((H_k * r).clip(max=0))
    return (m, mm)

  # make sure intermediate tables are all consistent
  def assert_state(A, H, C, M, MM):
    fail = False
    for k in xrange(K):
      C_i = (C == k)
      A_k = A[C_i,:]
      for j in xrange(N_):
        M_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[k, j], j]))
        MM_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[k, j], j]))
        assert M[k,j]  == M_c
        assert MM[k,j] == MM_c

  # sample initial haplotypes
  for k in xrange(K):
    A_k = A[(C == k),:]
    H[k,:] = sample_haplotype(M[k,:], MM[k,:], G[k,:,:], H[k,:], A_k)

  assert_state(A, H, C, M, MM)

  H_samples = np.zeros((20, K, N_))
  for iteration in xrange(100):
    print 'iteration', iteration
    #print '  - logP: {}'.format(logP)

    for i_p in xrange(M_):

      assert_state(A, H, C, M, MM)

      r_i = A[i_p,:]
      k_p = C[i_p]

      # matches with current assignment
      m, mm = get_hap_counts(r_i, H[k_p,:])
      
      # update betas by removing r_j from its current cluster
      M_p  = M[k_p,:] - m
      MM_p = MM[k_p,:] - mm

      assert (M_p >= 0).all()
      assert (MM_p >= 0).all()

      G_p = np.array(G[k_p,:,:])
      for j in np.nditer(np.nonzero(A[i_p,:] != 0)):
        G_p[j,:] = score_beta_site(M_p[j], MM_p[j])

      # set assignment to nil for now to resample hap
      C[i_p] = -1
      A_k = A[(C == k_p),:]
      H_p = sample_haplotype(M_p, MM_p, G_p, H[k_p,:], A_k)
      C[i_p] = k_p

      # score read under all haps
      scores = np.ones(K)
      for k in xrange(K):
        if k == k_p:
          m, mm = get_hap_counts(r_i, H_p)
          scores[k] = score_read(m, mm, M_p, MM_p)
        else:
          m, mm = get_hap_counts(r_i, H[k,:])
          scores[k] = score_read(m, mm, M[k,:], MM[k,:])
      assert_state(A, H, C, M, MM)

      # pick new cluster with prob proportional to scores under K
      # different betas
      scores = np.exp(scores - logsumexp(scores))
      assn = np.random.multinomial(1, scores)
      k_n = np.nonzero(assn == 1)[0][0]

      # update haplotypes with new assignment
      if k_n == k_p:
        pass
      else:
        # update previous haplotype to remove r_i
        assert_state(A, H, C, M, MM)
        M[k_p,:] = M_p
        MM[k_p,:] = MM_p
        G[k_p,:,:] = G_p
        H[k_p,:] = H_p
        C[i_p] = k_n
        # update next haplotype to add r_i
        m, mm = get_hap_counts(r_i, H[k_n,:])
        M[k_n,:]  = M[k_n,:] + m
        MM[k_n,:] = MM[k_n,:] + mm
        for j in np.nditer(np.nonzero(A[i_p,:] != 0)):
          G[k_n,j,:] = score_beta_site(M[k_n,j], MM[k_n,j])
        # resample updated haplotype
        A_k = A[(C == k_n),:]
        H[k_n,:] = sample_haplotype(M[k_n,:], MM[k_n,:], G[k_n,:,:], H[k_n,:], A_k)
        assert_state(A, H, C, M, MM)

    i = iteration % 20
    H_samples[i,:,:] = H

  print 'finished sampling'
  # convert ref from -1 to 0 so can compute sampled probability
  H_samples[H_samples == -1] = 0
  p_H = np.sum(H_samples,axis=0) / 20.
  print p_H

  h5_path = os.path.join(scratch_path, 'phased.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('H', data=H)
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

  # determine read assignment to haplotypes
  W = np.empty((M_,))
  W.fill(-1)

  logP, M, MM, S = score(A, H)

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

