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

(alpha, beta) = (1.0, 1.0)
(alpha, beta) = (1.0, 0.1)
(alpha, beta) = (1.0, 0.01)
K = 5
#K = 10

# normalization constant for the beta
beta_R = (
  gammaln(alpha + beta)
  - gammaln(alpha) +
  - gammaln(beta)
)

def get_hap_counts(r, H_k):
  m  = np.abs((H_k * r).clip(min=0))
  mm = np.abs((H_k * r).clip(max=0))
  return (m, mm)

def score_beta_site(m, mm):
  a = gammaln(m + alpha)
  b = gammaln(mm + beta)
  a_p = gammaln(mm + alpha)
  b_p = gammaln(m + beta)
  n = gammaln(m + mm + alpha + beta)
  logZ = logsumexp([a + b, a_p + b_p], axis=0)

  return a, b, a_p, b_p, n, logZ

def score(A, H, C):
  M_, N_ = A.shape
  logP = 0.0
  # match+mismatch counts for each site in each hap K
  M  = np.zeros((K, N_))
  MM = np.zeros((K, N_))
  # gamma counts
  G = np.zeros((K, N_, 6))
  # cluster scores with current assignment
  S = np.zeros((K, 1))

  # score assigned reads in each cluser by integrating over thetas
  for k in xrange(K):
    C_i = (C == k)
    A_k = A[C_i,:]
    for j in xrange(N_):
      M[k,j]  = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[k, j], j]))
      MM[k,j] = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[k, j], j]))
      G[k,j,:] = score_beta_site(M[k,j], MM[k,j])
    S[k] = (
      N_ * beta_R + 
      np.sum(G[k,:,0]) + 
      np.sum(G[k,:,1]) -
      np.sum(G[k,:,4])
    )
  logP = np.sum(S)

  return logP, M, MM, G, S

def get_initial_state(A):
  M_, N_ = A.shape

  # get inspired by some reads to initialize the hidden haplotypes
  #FIXME TODO
  H = np.ones((K, N_))
  C = np.random.choice(K, M_)
  C[:K] = np.arange(K)
  #print C
  #die

  ## FIXME remove
  ## seed with answer
  #H = np.array(A[:5,:])
  #C = np.arange(M_)
  #C = C % K

  return H, C

def score_read(m, mm, M, MM):
  logP = 0.

  mask = ((m > 0) | (mm > 0))
  N = M + MM
  # match component
  mlogP = np.sum((np.log(alpha + M[mask]) - np.log(alpha + beta + N[mask])) * m[mask])
  # mismatch component
  mmlogP = np.sum((np.log(beta + MM[mask]) - np.log(alpha + beta + N[mask])) * mm[mask])

  logP = mlogP + mmlogP

  return logP

def sample_haplotype_seed(r):
  N = r.shape[0]
  H = np.ones(N)
  m, mm = get_hap_counts(r, H)

  logP_flip = (
    (np.log(alpha) - np.log(alpha + beta)) * mm + 
    (np.log(beta) - np.log(alpha + beta)) * m
  )
  flip_mask = np.random.random(N) < np.exp(logP_flip)
  H[flip_mask] = -H[flip_mask]
  return H

def sample_haplotype(M, MM, G, H_p, A_k, seed=None):
  # must be populated with at least one read
  assert A_k.shape[0] > 0

  N = M.shape[0]
  H_n = np.zeros(N)

  def assert_hap_state(H):
    for j in xrange(N):
      M_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[j], j]))
      MM_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[j], j]))
      assert M[j]  == M_c
      assert MM[j] == MM_c
      assert (G[j,:] == score_beta_site(M_c, MM_c)).all()

  # FIXME TODO
  # can cheat and maybe only sample from the changed bits

  #assert_hap_state(H_p)
  logP_flip = (G[:,2] + G[:,3]) - G[:,5]
  # choose bits to flip
  p = np.random.random(N)
  flip_mask = p < np.exp(logP_flip)
  # flip selected bits for newly sampled haplotype
  H_n = np.array(H_p)
  H_n[flip_mask] = -H_p[flip_mask]
  # update sufficient statistics for loci that flipped
  fm = flip_mask
  MM[fm], M[fm] = M[fm], MM[fm]
  G[fm,0], G[fm,1], G[fm,2], G[fm,3] = G[fm,2], G[fm,3], G[fm,0], G[fm,1]
  #assert_hap_state(H_n)

  return H_n

def phase(scratch_path):
  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A = util.load_phase_inputs(h5_path)

  print 'loaded {} X {}'.format(len(bcodes), len(snps))

  M_, N_ = A.shape

  # hidden haplotypes
  H, C = get_initial_state(A)

  # initialize and save intermediate values for fast vectorized
  # computation
  logP, M, MM, G, S = score(A, H, C)

  # make sure intermediate tables are all consistent
  def assert_state(A, H, C, M, MM, G):
    fail = False
    for k in xrange(K):
      C_i = (C == k)
      A_k = A[C_i,:]
      for j in xrange(N_):
        M_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[k, j], j]))
        MM_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[k, j], j]))
        assert M[k,j]  == M_c
        assert MM[k,j] == MM_c
        assert (G[k,j,:] == score_beta_site(M_c, MM_c)).all()

  assert_state(A, H, C, M, MM, G)

  # sample initial haplotypes
  for k in xrange(K):
    A_k = A[(C == k),:]
    H[k,:] = sample_haplotype(M[k,:], MM[k,:], G[k,:,:], H[k,:], A_k)

  # compute seed gammas for empty hap
  G_seed = np.zeros((N_, 6))
  G_seed[:,0], G_seed[:,1], G_seed[:,2], G_seed[:,3], G_seed[:,4], G_seed[:,5] = \
    score_beta_site(np.zeros(N_), np.zeros(N_))
 
  num_iterations = 400
  num_samples = 50
  sample_step = 1

  #num_iterations = 10000
  #num_samples = 50
  #sample_step = 100

  assert sample_step * num_samples < num_iterations, \
    "{} iterations is not enough for {} samples with step {}".format(
      num_iterations,
      num_samples,
      sample_step,
    )

  sidx = 0
  H_samples = np.zeros((num_samples, K, N_))
  C_samples = np.zeros((num_samples, M_))
  for iteration in xrange(num_iterations):
    print 'iteration', iteration
    if iteration % 100 == 0:
      print 'iteration', iteration

    # save sample
    if iteration % sample_step == 0:
      H_samples[sidx,:,:] = H
      C_samples[sidx,:] = C
      sidx = (sidx + 1) % num_samples
    #print 'H', H
    #print 'C', C

    for i_p in xrange(M_):
      #print 'read', i_p
      #if i_p > 100:
      #  print 'early exit!'
      #  sys.exit(0)

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
      if A_k.shape[0] == 0:
        H_p = sample_haplotype_seed(r_i)
      else:
        H_p = sample_haplotype(M_p, MM_p, G_p, H[k_p,:], A_k)
      C[i_p] = k_p

      # score read under all haps
      scores = np.ones(K)
      for k in xrange(K):
        # if hap k is currently empty, then resample under this seed r_i
        if not (C == k).any():
          H[k,:] = sample_haplotype_seed(r_i)
          G[k,:,:] = np.array(G_seed)
        if k == k_p:
          m, mm = get_hap_counts(r_i, H_p)
          scores[k] = score_read(m, mm, M_p, MM_p)
        else:
          m, mm = get_hap_counts(r_i, H[k,:])
          scores[k] = score_read(m, mm, M[k,:], MM[k,:])

      # pick new cluster with prob proportional to scores under K
      # different betas
      scores = np.exp(scores - logsumexp(scores))
      assn = np.random.multinomial(1, scores)
      k_n = np.nonzero(assn == 1)[0][0]

      # resample since stayed
      if k_n == k_p:
        A_k = A[(C == k_n),:]
        H[k_n,:] = sample_haplotype(M[k_n,:], MM[k_n,:], G[k_n,:,:], H[k_n,:], A_k)
      # update haplotypes with new assignment
      else:
        #print 'move taken (k_p, k_n)', (k_p, k_n)
        #print scores
        #die
        # update previous haplotype to remove r_i
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

  assert_state(A, H, C, M, MM, G)

  print 'finished sampling'
  # convert ref from -1 to 0 so can compute sampled probability
  H_samples[H_samples == -1] = 0
  p_H = np.sum(H_samples,axis=0) / num_samples
  p_C = np.zeros((M_,K))
  for k in xrange(K):
    p_C[:,k] = 1.* np.sum(C_samples == k,axis=0) / num_samples
  #for i in xrange(num_samples):
  #  print 'H sample'
  #  print H_samples[i,:,:]
  #print C_samples

  #print p_H
  #print p_C

  h5_path = os.path.join(scratch_path, 'phased.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('p_H', data=p_H)
  h5f.create_dataset('p_C', data=p_C)
  h5f.close()

def make_outputs(scratch_path):
  inputsh5_path = os.path.join(scratch_path, 'inputs.h5')
  phaseh5_path = os.path.join(scratch_path, 'phased.h5')
  snps, bcodes, A = util.load_phase_inputs(inputsh5_path)
  h5f = h5py.File(phaseh5_path, 'r')
  p_H = np.array(h5f['p_H'])
  p_C = np.array(h5f['p_C'])
  h5f.close()
  M_, N_ = A.shape
  idx_rid_map = dict(list(enumerate(bcodes)))

  print 'loaded {} X {}'.format(len(bcodes), len(snps))

  # round probs to get haplotype matrix
  K_, N_ = p_H.shape
  H = np.zeros((K_, N_))
  H[p_H >= 0.5] = 1
  H[p_H < 0.5] = -1

  # determine read assignment to haplotypes
  W = np.empty((M_,))
  W.fill(-1)
  for i in xrange(M_):
    assn = np.argmax(p_C[i,:])
    if p_C[i,assn] > 0.8:
      W[i] = assn
    else:
      pass

  outdir_path = os.path.join(scratch_path, 'bins')
  util.mkdir_p(outdir_path)
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

  out_path = os.path.join(scratch_path, 'bins', 'clusters.p')
  util.write_pickle(out_path, clusters_map)
  print 'assigned barcodes', sum((W != -1))
  print 'unassigned barcodes', sum((W == -1))

