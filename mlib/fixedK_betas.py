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
import debug

debug_rid = 1928

np.random.seed(0)

(alpha, beta) = (1.0, 1.0)
(alpha, beta) = (1.0, 0.1)
(alpha, beta) = (1.0, 0.01)
(alpha, beta) = (1.0, 0.1)
#K = 5
K = 10

#num_iterations = 2000
num_iterations = 800
#num_iterations = 400
num_samples = 50
sample_step = 2
sample_step = 1

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

#@profile
def phase(scratch_path):
  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A = util.load_phase_inputs(h5_path)

  print 'loaded {} X {}'.format(len(bcodes), len(snps))
  bcodes, A = util.subsample_reads(bcodes, A, lim=5000)
  print '  - subsample to {} X {}'.format(len(bcodes), len(snps))

  # FIXME change to assert
  print 'number nonzero', np.sum(np.any((A != 0), axis=1))

  M_, N_ = A.shape

  # hidden haplotypes
  H, C = util.get_initial_state(A, K)

  phaseh5_path = os.path.join(scratch_path, 'phased.h5')
  h5f = h5py.File(phaseh5_path, 'r')
  H_samples = np.array(h5f['H_samples'])
  C_samples = np.array(h5f['C_samples'])
  h5f.close()
  H = np.array(H_samples[-1,:,:])
  C = np.array(C_samples[-1,:], dtype=int)

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
 
  assert sample_step * num_samples < num_iterations, \
    "{} iterations is not enough for {} samples with step {}".format(
      num_iterations,
      num_samples,
      sample_step,
    )

  sidx = 0
  H_samples = np.zeros((num_iterations, K, N_))
  C_samples = np.zeros((num_iterations, M_))
  for iteration in xrange(num_iterations):

    print 'iteration', iteration
    if iteration % 50 == 0:
      print 'iteration', iteration

    # save all samples from each iteration for now
    H_samples[sidx,:,:] = H
    C_samples[sidx,:] = C
    sidx = (sidx + 1) % num_iterations

    for i_p in xrange(M_):

      if i_p != debug_rid:
        continue

      r_i = A[i_p,:]

      # FIXME remove
      # skip empty reads that made their way into genotype matrix from bug
      # in mkinputs
      if (r_i == 0).all():
        continue

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
      print 'scoreslog ', scores
      scores = np.exp(scores - logsumexp(scores))
      print 'scores p', scores
      assn = np.random.multinomial(1, scores)
      k_n = np.nonzero(assn == 1)[0][0]
      print 'prev, next', (k_p, k_n)

      mismatches = np.zeros(K)
      for _k in xrange(K):
        mismatches[_k] = np.sum(r_i * H[_k,:] < 0)
      print 'mismatches', mismatches
      die

      # resample since stayed
      if k_n == k_p:
        #A_k = A[(C == k_n),:]
        A_k = None
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
        A_k = None
        #A_k = A[(C == k_n),:]
        H[k_n,:] = sample_haplotype(M[k_n,:], MM[k_n,:], G[k_n,:,:], H[k_n,:], A_k)

  assert_state(A, H, C, M, MM, G)

  print 'finished sampling'
  print 'num samples', num_samples

  h5_path = os.path.join(scratch_path, 'phased.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('H_samples', data=H_samples)
  h5f.create_dataset('C_samples', data=C_samples)
  h5f.close()

def make_outputs(scratch_path):
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

  # convert ref from -1 to 0 so can compute sampled probability
  H_samples[H_samples == -1] = 0

  def get_empirical_probs(sidx):

    eidx = sidx + num_samples * sample_step
    assert eidx < num_iterations
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

  p_C, p_H, H = get_empirical_probs(0)
  W0 = get_assignments(p_C)
  print 'number of reads not sticking...', np.sum(W0 == -1)
  print 'total reads that are in the same place 80% of time', \
    np.sum(W0 != -1)

  p_C, p_H, H = get_empirical_probs(200)
  W1 = get_assignments(p_C)
  print 'total reads that are in the same place 80% of time', \
    np.sum(W1 != -1), np.sum(W0 != W1)
  print 'number of reads not sticking...', np.sum(W1 == -1)

  p_C, p_H, H = get_empirical_probs(600)
  W2 = get_assignments(p_C)
  print 'total reads that are in the same place 80% of time', \
    np.sum(W2 != -1), np.sum(W2 != W1)
  print 'number of reads not sticking...', np.sum(W2 == -1)
  print p_C

  debugdir_path = os.path.join(scratch_path, 'debug')

  util.mkdir_p(debugdir_path)

  #_, W = util.get_initial_state(A)

  def get_bcodes_from_rids(rids):
    bcodes = set()
    for rid in np.nditer(rids):
      rid = int(rid)
      bcodes.add(idx_rid_map[rid])
    return bcodes
  
  sample_sizes = num_samples * sample_step
  num_start_points = (num_iterations - sample_sizes - 100) / sample_sizes
  print 'num different starting sample points', num_start_points
  E = np.zeros((num_start_points, K))
  F = np.zeros((num_start_points))
  W_prev = np.empty((M_,))
  W_prev.fill(-1)
  for i, sidx in enumerate(
    xrange(100,num_iterations - sample_sizes, sample_sizes)
  ):
    p_C, p_H, H = get_empirical_probs(sidx)
    W = get_assignments(p_C)

    #H = np.array(H_samples[sidx,:,:])
    #H[H == 0] = -1
    #W = C_samples[sidx,:]

    F[i] = np.sum(W != W_prev)
    W_prev = np.array(W)
    for k in xrange(K):
      A_k = A[(W == k),:]
      _, entr, _ = debug.plot_entropy(
        k,
        debugdir_path,
        A_k,
        debug=False,
      )
      num_hsites = np.sum(entr > 0.5)
      E[i,k] = num_hsites
  print 'total entropy over starting points'
  print E
  print 'entropy summed over all K haps'
  print np.sum(E, axis=1)
  print 'number of changes in read assignment from previous samples'
  print F

  # check final read dest is the best place for it
  p_C, p_H, H = get_empirical_probs(600)
  W = get_assignments(p_C)

  #sidx = 600
  #H = np.array(H_samples[sidx,:,:])
  #H[H == 0] = -1
  #W = C_samples[sidx,:]

  print 'checking final read resting spots'
  logP, M, MM, G, S = score(A, H, W)

  H = H_samples[750,:,:]
  num_misplaced = 0
  np.set_printoptions(threshold=np.nan,linewidth=900)
  for i in xrange(M_):
    r = np.array(A[i,:])
    r_n = np.array(A[i,:])
    r_n[r>0] = 1
    r_n[r<0] = -1
    r = r_n
    ak = int(W[i])
    mismatches = np.zeros(K)
    scores = np.zeros(K)
    m, mm = get_hap_counts(r, H[ak,:])
    M_p = M[ak,:] - m
    MM_p = MM[ak,:] - mm

    G_p = np.array(G[ak,:,:])
    for j in np.nditer(np.nonzero(r != 0)):
      G_p[j,:] = score_beta_site(M_p[j], MM_p[j])
    H_p = sample_haplotype(M_p, MM_p, G_p, H[ak,:], None)

    for _k in xrange(K):
      mismatches[_k] = np.sum(r * H[_k,:] < 0)
      if _k == ak:
        m, mm = get_hap_counts(r, H_p)
        scores[_k] = score_read(m, mm, M_p, MM_p)
      else:
        m, mm = get_hap_counts(r, H[_k,:])
        scores[_k] = score_read(m, mm, M[_k,:], MM[_k,:])
    ck = np.argmin(mismatches)
    #if (
    #  mismatches[ak] != mismatches[ck] and
    #  W[i] == C_samples[sidx+1,i]
    #):
    if W[i] == 5 and i == 1928:
      num_misplaced += 1
      print 'best match is not final resting spot for rid', i
      print '  num sites', np.sum(np.abs(r))
      print '  mismatches', mismatches
      print '  scores', scores
      print '  - computed k', np.argmin(mismatches)
      print '  - assigned k', ak

      print 'assigned cluster', ak
      m, mm = get_hap_counts(r, H[ak,:])
      print ' M  at mismatch vals', M_p[mm > 0]
      print ' MM at mismatch vals', MM_p[mm > 0]
      print 
      print 'computed cluster', ck
      m, mm = get_hap_counts(r, H[ck,:])
      print ' M  at mismatch vals', M[ck, mm > 0]
      print ' MM at mismatch vals', MM[ck, mm > 0]
      print 

      print 'renormalized scores'
      print np.exp(scores - logsumexp(scores))

      print 'read assn probs past first 100:', set(C_samples[100:,i])
      print 'sample assignments'
      print C_samples[:,i]
      print
      die

    else:
      pass

  print 'number of reads misplaced', num_misplaced

  outdir_path = os.path.join(scratch_path, 'bins')
  util.mkdir_p(outdir_path)
  clusters_map = {}
  for k in xrange(K):
    sel_rids = np.nonzero(W == k)[0]
    if len(sel_rids) > 0:
      bcode_set = get_bcodes_from_rids(sel_rids)
    else:
      bcode_set = set()
    clusters_map[k] = bcode_set
    out_path = os.path.join(outdir_path, '{}.bin.txt'.format(k)) 
    with open(out_path, 'w') as fout:
      for bcode in bcode_set:
        fout.write('{}\n'.format(bcode))

  h5_path = os.path.join(scratch_path, 'bins', 'clusters.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('H', data=H)
  h5f.create_dataset('W', data=W)
  h5f.close()

  out_path = os.path.join(scratch_path, 'bins', 'clusters.p')
  util.write_pickle(out_path, clusters_map)

  print 'total haplotype positions', K * N_
  print '  - assigned', np.sum(np.abs(H))

  print 'assigned barcodes', sum((W != -1))
  print 'unassigned barcodes', sum((W == -1))

