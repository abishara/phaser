import os
import sys
import numpy as np
import math
import random
from scipy.misc import logsumexp
from scipy.special import gammaln
from collections import defaultdict, Counter
import h5py
import pysam

import util
import hyper_params as hp

np.random.seed(0)
random.seed(0)

(alpha, beta) = (1.0, 0.1)
(alpha, beta) = (1.0, 0.01)
DP_alpha = 1

num_iterations = 800
#num_iterations = 400
num_samples = 50
sample_step = 2

# normalization constant for the beta
beta_R = (
  gammaln(alpha + beta)
  - gammaln(alpha) +
  - gammaln(beta)
)

# compute seed gammas for empty haplotype cluster
G_seed = None

def get_hap_counts(r, H_k):
  m  = np.abs((H_k * r).clip(min=0))
  mm = np.abs((H_k * r).clip(max=0))
  return (m, mm)

beta_cache = {}
def score_beta_site(m, mm):
  if m.shape == () and (m, mm) in beta_cache:
    return beta_cache[(m, mm)]
  a = gammaln(m + alpha)
  b = gammaln(mm + beta)
  a_p = gammaln(mm + alpha)
  b_p = gammaln(m + beta)
  n = gammaln(m + mm + alpha + beta)
  logZ = logsumexp([a + b, a_p + b_p], axis=0)
  if m.shape == ():
    beta_cache[(m, mm)] = (a, b, a_p, b_p, n, logZ)
  return a, b, a_p, b_p, n, logZ

def score(A, H, C):
  M_, N_ = A.shape
  K, _ = H.shape
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

  N = M.shape[0]
  H_n = np.zeros(N)

  def assert_hap_state(H):
    assert A_k.shape[0] > 0
    for j in xrange(N):
      M_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[j], j]))
      MM_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[j], j]))
      assert M[j]  == M_c
      assert MM[j] == MM_c
      assert (G[j,:] == score_beta_site(M_c, MM_c)).all()

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

#--------------------------------------------------------------------------
# merge-split MH move
#--------------------------------------------------------------------------
def merge_split(A, H, C, M, MM, G, S):

  M_, N_ = A.shape
  K, _ = H.shape

  # draw to reads
  i = random.randint(0,N_-1)
  j = random.randint(0,N_-1)
  j = (i-1)%(N_-1) if i == j else j
  if j < i:
    i,j = j,i

  # FIXME direct hints for split operation.  don't split unless there's
  # some disagreement between the reads...

  s_m = (C == C[i]) | (C == C[j])
  sel_rids = np.nonzero(s_m)[0]

  #print 'i,j', i,j

  r_i = A[i,:]
  r_j = A[i,:]
  c_i = C[i]
  c_j = C[j]

  i_s = np.nonzero(sel_rids == i)[0]
  j_s = np.nonzero(sel_rids == j)[0]

  # determine launch state
  A_s = A[s_m,:]
  num_s, _ = A_s.shape

  m = np.zeros(K)
  m[c_i] = 1
  m[c_j] = 1
  m = np.ma.make_mask(m)

  H_orig = H[m,:]
  C_orig = C[s_m]
  C_orig[(C_orig == c_i)] = 0
  C_orig[(C_orig == c_j)] = 1
  
  # place each read to the closer of read i or read j
  C_s = np.zeros(num_s, dtype=np.int)
  for k_s in xrange(num_s):
    r_k = A_s[k_s]
    m_i, mm_i = get_hap_counts(r_i, r_k)
    m_j, mm_j = get_hap_counts(r_j, r_k)
    C_s[k_s] = 0 if np.sum(mm_i) < np.sum(mm_j) else 1
  C_s[i_s] = 0
  C_s[j_s] = 1

  # initial haps
  H_s = np.zeros((2, N_))
  H_s[0,:] = H[C[i],:]
  H_s[1,:] = H[C[j],:]

  # intermediate restricted gibbs scan to get sensible split
  _, M_s, MM_s, G_s, S_s = score(A_s, H_s, C_s)

  # merge score
  H_merge = np.ones((1, N_))
  C_merge = np.zeros(num_s, dtype=np.int)

  _, t_M, t_MM, t_G, t_S = score(A_s, H_merge, C_merge)
  H_merge[0,:] = sample_haplotype(t_M[0,:], t_MM[0,:], t_G[0,:,:], H_merge[0,:], None)
  logPs_merge, t_M, t_MM, t_G, t_S = score(A_s, H_merge, C_merge)

  # FIXME technically reads i and j should be fixed during the gibbs scan
  for local_iteration in xrange(3):
    H_s, C_s, _ = gibbs_scan(A_s, H_s, C_s, M_s, MM_s, G_s, S_s)
  H_launch = np.array(H_s)
  C_launch = np.array(C_s)

  acc_log_prior = gammaln(np.sum(C_s) ==0)

  # split operation
  if C[i] == C[j]:
    # transition from launch to final split state
    H_s, C_s, trans_logP = gibbs_scan(A_s, H_s, C_s, M_s, MM_s, G_s, S_s)
    logPs_split, _, _, _, _ = score(A_s, H_s, C_s)

    n_ci = np.sum(C_s == 0)
    n_cj = np.sum(C_s == 1)
    acc_log_prior = (
      np.log(DP_alpha) + 
      gammaln(n_ci) + gammaln(n_cj) - 
      gammaln(n_ci + n_cj)
    )
    acc_log_ll = np.sum(logPs_split) - np.sum(logPs_merge)
    acc_log_trans = -trans_logP

    acc = min(1, np.exp(acc_log_prior + acc_log_ll + acc_log_trans))
    if random.random() <= acc:

      # update split state to global state
      C_s[(C_s == 0)] = c_i
      C_s[(C_s == 1)] = K
      C[s_m] = C_s
      H = np.vstack([H, H_s[1]])
      H[c_i,:] = H_s[0]
      # update cached values and append to end for new cluster
      M[c_i,:] = M_s[0,:]
      MM[c_i,:] = MM_s[0,:]
      G[c_i,:,:] = G_s[0,:,:]
      S[c_i,:] = S_s[0,:]

      M  = np.vstack([M, np.zeros(N_)])
      MM = np.vstack([MM, np.zeros(N_)])
      G_p = G
      G = np.zeros((K+1,N_,6))
      G[:K,:,:] = G_p
      S = np.vstack([S, np.zeros(1)])

      M[K,:] = M_s[1,:]
      MM[K,:] = MM_s[1,:]
      G[K,:,:] = G_s[1,:,:]
      S[K,:] = S_s[1,:]

    else:
      pass

  # merge operation
  else:
    #print 'merge operation'

    # transition from launch to original split state
    H_s, C_s, trans_logP = gibbs_scan(
      A_s, H_launch, C_launch, M_s, MM_s, G_s, S_s,
      trans_path=C_orig,
    )
    #print 'C_orig', C_orig
    #print 'C_launch', C_launch
    #print 'trans_logP', trans_logP
    #die

    logPs_split, _, _, _, _ = score(A_s, H_orig, C_orig)
    n_ci = np.sum(C_orig == 0)
    n_cj = np.sum(C_orig == 1)
    acc_log_prior = (
      gammaln(n_ci + n_cj)  -
      (gammaln(n_ci) + gammaln(n_cj)) - 
      np.log(DP_alpha)
    )
    acc_log_ll = np.sum(logPs_merge) - np.sum(logPs_split)
    acc_log_trans = trans_logP
    #print 'acc_log_prior', acc_log_prior
    #print 'logPs_merge', logPs_merge
    #print 'logPs_split', logPs_split
    #print 'acc_log_ll', acc_log_ll
    #print 'trans_logP', trans_logP

    acc = min(1, np.exp(acc_log_prior + acc_log_ll + acc_log_trans))

    if random.random() <= acc:
      # update merge state in global state
      C[s_m] = c_i
      m = np.ones(K)
      m[c_j] = 0
      m = np.ma.make_mask(m)

      # assigned resampled haplotype for new merged cluster
      H[c_i] = H_merge[0,:]

      # update cached values for new cluster
      M[c_i,:] = t_M[0,:]
      MM[c_i,:] = t_MM[0,:]
      G[c_i,:,:] = t_G[0,:,:]
      S[c_i,:] = t_S[0,:]

      # drop old haplotype cluster c_j
      H = H[m,:]
      M = M[m,:]
      MM = MM[m,:]
      G = G[m,:,:]
      S = S[m,:]
      # shift index assignments down
      C[C > c_j] -= 1

      #print 'new state'
      #print 'mask', m
      #print 'H[3,:] ', H[3,:]
      #print 'M[3,:] ', M[3,:]
      #print 'MM[3,:]', MM[3,:]
      #print 'reads'
      #print 
      #print A[(C == 3)]

    else:
      pass

  return  H, C, M, MM, G, S

#--------------------------------------------------------------------------
# gibbs scan
#--------------------------------------------------------------------------
def gibbs_scan(A, H, C, M, MM, G, S, trans_path=None):
  trans_logP = 0.
  M_, N_ = A.shape
  K, _ = H.shape

  if trans_path is not None:
    assert trans_path.shape[0] == M_, \
      "transition path not fully specified, wrong dimension"
  for i_p in xrange(M_):
  
    r_i = A[i_p,:]
    k_p = C[i_p]
    k_fixed = None if trans_path is None else trans_path[i_p]
  
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
    if not (C == k_p).any():
      H_p = sample_haplotype_seed(r_i)
    else:
      A_k = None
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
    log_scores = scores - logsumexp(scores)
    scores = np.exp(log_scores)
    assn = np.random.multinomial(1, scores)
    # take fixed transition path if specified
    if k_fixed is not None:
      k_n = k_fixed
    else:
      k_n = np.nonzero(assn == 1)[0][0]
    trans_logP += log_scores[k_n]
  
    # resample since stayed
    if k_n == k_p:
      #A_k = A[(C == k_n),:]
      A_k = None
      H[k_n,:] = sample_haplotype(M[k_n,:], MM[k_n,:], G[k_n,:,:], H[k_n,:], A_k)
    # update haplotypes with new assignment
    else:
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
      A_k = None
      #A_k = A[(C == k_n),:]
      H[k_n,:] = sample_haplotype(M[k_n,:], MM[k_n,:], G[k_n,:,:], H[k_n,:], A_k)

  return H, C, trans_logP

#--------------------------------------------------------------------------
# phasing
#--------------------------------------------------------------------------
def phase(scratch_path):
  global G_seed

  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A = util.load_phase_inputs(h5_path)

  print 'loaded {} X {}'.format(len(bcodes), len(snps))
  bcodes, A = util.subsample_reads(bcodes, A, lim=5000)
  print '  - subsample to {} X {}'.format(len(bcodes), len(snps))

  # FIXME change to assert
  print 'number nonzero', np.sum(np.any((A != 0), axis=1))

  M_, N_ = A.shape

  # hidden haplotypes
  H, C = util.get_initial_state(A, hp.K)

  # initialize and save intermediate values for fast vectorized computation
  logP, M, MM, G, S = score(A, H, C)

  # sample initial haplotypes
  for k in xrange(hp.K):
    A_k = A[(C == k),:]
    H[k,:] = sample_haplotype(M[k,:], MM[k,:], G[k,:,:], H[k,:], A_k)

  # make sure intermediate tables are all consistent
  def assert_state(A, H, C, M, MM, G):
    fail = False
    K_, _ = H.shape
    for k in xrange(K_):
      C_i = (C == k)
      A_k = A[C_i,:]
      for j in xrange(N_):
        M_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == H[k, j], j]))
        MM_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -H[k, j], j]))
        assert M[k,j]  == M_c
        assert M[k,j]  == M_c
        assert MM[k,j] == MM_c
        assert (G[k,j,:] == score_beta_site(M_c, MM_c)).all()

  # compute seed gammas for empty haplotype cluster
  G_seed = np.zeros((N_, 6))
  G_seed[:,0], G_seed[:,1], G_seed[:,2], G_seed[:,3], G_seed[:,4], G_seed[:,5] = \
    score_beta_site(np.zeros(N_), np.zeros(N_))

  score_beta_site(np.zeros(N_), np.zeros(N_))

  assert_state(A, H, C, M, MM, G)

  assert sample_step * num_samples < num_iterations, \
    "{} iterations is not enough for {} samples with step {}".format(
      num_iterations,
      num_samples,
      sample_step,
    )

  sidx = 0
  H_samples = []
  C_samples = []
  #H_samples = np.zeros((num_iterations, K, N_))
  #C_samples = np.zeros((num_iterations, M_))
  for iteration in xrange(num_iterations):

    print 'iteration', iteration
    if iteration % 50 == 0:
      print 'iteration', iteration

    #H, C, _ = gibbs_scan(A, H, C, M, MM, G, S)

    assert_state(A, H, C, M, MM, G)
    H, C, M, MM, G, S =  merge_split(A, H, C, M, MM, G, S)
    assert_state(A, H, C, M, MM, G)

    H_samples.append(H)
    C_samples.append(C)
    ## save all samples from each iteration for now
    #H_samples[sidx,:,:] = H
    #C_samples[sidx,:] = C
    #sidx = (sidx + 1) % num_iterations

  assert_state(A, H, C, M, MM, G)

  print 'finished sampling'
  print 'num samples', num_samples

  h5_path = os.path.join(scratch_path, 'phased.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('H_samples', data=H_samples)
  h5f.create_dataset('C_samples', data=C_samples)
  h5f.close()

def make_outputs(inbam_path, scratch_path):
  inputsh5_path = os.path.join(scratch_path, 'inputs.h5')
  phaseh5_path = os.path.join(scratch_path, 'phased.h5')
  snps, bcodes, A = util.load_phase_inputs(inputsh5_path)
  h5f = h5py.File(phaseh5_path, 'r')
  H_samples = np.array(h5f['H_samples'])
  C_samples = np.array(h5f['C_samples'])
  h5f.close()

  print 'loaded {} X {}'.format(len(bcodes), len(snps))
  bcodes, A = util.subsample_reads(bcodes, A, lim=5000)
  print '  - subsample to {} X {}'.format(len(bcodes), len(snps))

  idx_rid_map = dict(list(enumerate(bcodes)))
  idx_snp_map = dict(list(enumerate(snps)))
  M_, N_ = A.shape
  K = hp.K

  # convert ref from -1 to 0 so can compute sampled probability
  H_samples[H_samples == -1] = 0

  def get_empirical_probs():
    eidx = num_iterations - 1
    sidx = eidx - num_samples * sample_step
    assert eidx < num_iterations and sidx > 0
    p_H = np.sum(H_samples[sidx:eidx:sample_step,:,:],axis=0) / num_samples
    p_C = np.zeros((M_,K))
    for k in xrange(K):
      p_C[:,k] = 1.* np.sum(C_samples[sidx:eidx:sample_step,:] == k,axis=0) / num_samples

    # round probs to get haplotype matrix
    K_, N_ = p_H.shape
    H = np.zeros((K_, N_))
    H[p_H >= 0.95] = 1
    H[p_H < 0.05] = -1
    return p_C, p_H, H

  def get_assignments(p_C):
    # determine read assignment to haplotypes
    W = np.empty((M_,))
    W.fill(-1)
    for i in xrange(M_):
      assn = np.argmax(p_C[i,:])
      if p_C[i,assn] > 0.8:
        W[i] = assn
      else:
        pass
    return W

  p_C, p_H, H = get_empirical_probs()
  W = get_assignments(p_C)

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

  print 'total haplotype positions', K * N_
  print '  - assigned', np.sum(np.abs(H))
  print 'assigned barcodes', sum((W != -1))
  print 'unassigned barcodes', sum((W == -1))

