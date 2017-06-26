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

(alpha, beta) = (1.0, 1.0)
(alpha, beta) = (0.1, 0.1)
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

#--------------------------------------------------------------------------
# temporary stuff to score simulation moves
#--------------------------------------------------------------------------
def score_assignment(K, C, labels_map):

  def pp(cnt):
    for k, v in cnt.most_common():
      print '{}:{},'.format(k,v),
    print
    return

  accs = []
  for k in xrange(K):
    members = np.nonzero(C == k)[0]
    total = len(members)
    cnt = Counter(map(lambda(m):labels_map[m], members))
    acc = -1 if total == 0 else 1.* cnt.most_common()[0][1] / total
    accs.append(acc)
    print 'cluster', k 
    pp(cnt)
  return sorted(accs)
    
#--------------------------------------------------------------------------
# helpers
#--------------------------------------------------------------------------

def get_ref_alt(r):
  m  = np.abs(r.clip(min=0))
  mm = np.abs(r.clip(max=0))
  return (m, mm)

def get_matches(r1, r2):
  m  = np.abs((r1 * r2).clip(min=0))
  mm = np.abs((r1 * r2).clip(max=0))
  return (m, mm)

beta_cache = {}
def score_beta_site(m, mm):
  if m.shape == () and (m, mm) in beta_cache:
    return beta_cache[(m, mm)]
  a = gammaln(m + alpha)
  b = gammaln(mm + beta)
  n = gammaln(m + mm + alpha + beta)
  if m.shape == ():
    beta_cache[(m, mm)] = (a, b, n)
  return a, b, n

def score(A, K, C):
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
    C_i = (C == k)
    A_k = A[C_i,:]
    for j in xrange(N_):
      M[k,j]  = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == 1, j]))
      MM[k,j] = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -1, j]))
      G[k,j,:] = score_beta_site(M[k,j], MM[k,j])
    S[k] = (
      N_ * beta_R + 
      np.sum(G[k,:,0]) + 
      np.sum(G[k,:,1]) -
      np.sum(G[k,:,2])
    )
  logP = np.sum(S)

  return logP, M, MM, G, S

def score_read(m, mm, M, MM):
  logP = 0.

  mask0 = (m > 0)
  mask1 = (mm > 0)
  N = M + MM
  # match component
  mlogP = np.sum((np.log(alpha + M[mask0]) - np.log(alpha + beta + N[mask0])) * m[mask0])
  # mismatch component
  mmlogP = np.sum((np.log(beta + MM[mask1]) - np.log(alpha + beta + N[mask1])) * mm[mask1])

  logP = mlogP + mmlogP

  return logP

#--------------------------------------------------------------------------
# merge-split MH move
#--------------------------------------------------------------------------
#@profile
def merge_split(A, K, C, M, MM, G, S, 
  true_labels_map, bcodes_map, snps_map):

  M_, N_ = A.shape

  # draw to reads
  i = random.randint(0,M_-1)
  j = random.randint(0,M_-1)
  j = (i-1)%(N_-1) if i == j else j
  if j < i:
    i,j = j,i

  # FIXME direct hints for split operation.  don't split unless there's
  # some disagreement between the reads...

  s_m = (C == C[i]) | (C == C[j])
  sel_rids = np.nonzero(s_m)[0]

  #print 'i,j', i,j

  r_i = A[i,:]
  r_j = A[j,:]
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

  C_orig = C[s_m]
  m0 = (C_orig == c_i)
  m1 = (C_orig == c_j)
  C_orig[m0] = 0
  C_orig[m1] = 1
  
  # place each read to the closer of read i or read j
  C_s = np.zeros(num_s, dtype=np.int)

  ri_label = true_labels_map[i]
  rj_label = true_labels_map[j]
  bad_assn = 0
  good_assn = 0
  for rid, k_s in zip(sel_rids, range(num_s)):
    label = true_labels_map[rid]
    r_k = A_s[k_s]
    m_i, mm_i = map(np.sum, get_matches(r_i, r_k))
    m_j, mm_j = map(np.sum, get_matches(r_j, r_k))
    assn = 0 if m_i - mm_i > m_j - mm_j else 1
    C_s[k_s] = assn
    if label in [ri_label, rj_label] and not (
      (label == ri_label and assn == 0) or
      (label == rj_label and assn == 1)
    ):
      #if (label == ri_label) and mm_i > 0:
      #  print 'wtf'
      #  print 'mismatched label'
      #  print '  - assn', assn, label
      #  print '  - (m_i, mm_i)', m_i, mm_i
      #  print '  - (m_j, mm_j)', m_j, mm_j
      #  print 'bcode', bcodes_map[rid]
      #  die
      bad_assn += 1
    elif label in [ri_label, rj_label]:
      good_assn += 1
  #print 'c_i:{} r_i label:{}'.format(c_i, ri_label)
  #print 'c_j:{} r_j label:{}'.format(c_j, rj_label)
  #print '  pre local gibbs good/bad assn ', good_assn, bad_assn

  legit_split = (ri_label != rj_label)
     
  C_s[i_s] = 0
  C_s[j_s] = 1
  fixed_rids = set([i_s[0], j_s[0]])

  # merge score
  C_merge = np.zeros(num_s, dtype=np.int)
  logPs_merge, t_M, t_MM, t_G, t_S = score(A_s, 1, C_merge)

  # intermediate restricted gibbs scan to get sensible split
  _, M_s, MM_s, G_s, S_s = score(A_s, 2, C_s)

  # FIXME technically reads i and j should be fixed during the gibbs scan
  #print 'local gibbs scan'
  for local_iteration in xrange(3):
    C_s, _ = gibbs_scan(A_s, 2, C_s, M_s, MM_s, G_s, S_s,
      fixed_rids=fixed_rids)
  C_launch = np.array(C_s)

  # FIXME remove
  #-----------------------------------------------------------------------
  bad_assn = 0
  good_assn = 0
  for rid, k_s in zip(sel_rids, range(num_s)):
    label = true_labels_map[rid]
    assn = C_launch[k_s]
    if label in [ri_label, rj_label] and not (
      (label == ri_label and assn == 0) or
      (label == rj_label and assn == 1)
    ):
      bad_assn += 1
    elif label in [ri_label, rj_label]:
      good_assn += 1
  #print '  post local gibbs good/bad assn ', good_assn, bad_assn
  #-----------------------------------------------------------------------

  taken = False
  # split operation
  if C[i] == C[j]:
    # transition from launch to final split state
    C_s, trans_logP = gibbs_scan(A_s, 2, C_s, M_s, MM_s, G_s, S_s,
      fixed_rids=fixed_rids)
    logPs_split, _, _, _, _ = score(A_s, 2, C_s)

    n_ci = np.sum(C_s == 0)
    n_cj = np.sum(C_s == 1)
    #assert min(n_ci, n_cj) > 0

    acc_log_prior = (
      np.log(DP_alpha) + 
      gammaln(n_ci) + gammaln(n_cj) - 
      gammaln(n_ci + n_cj)
    )
    acc_log_ll = np.sum(logPs_split) - np.sum(logPs_merge)
    acc_log_trans = -trans_logP

    acc = min(1, np.exp(acc_log_prior + acc_log_ll + acc_log_trans))
    if random.random() <= acc and min(n_ci, n_cj) > 0:
      taken = True
      print 'splitting cluster', C[i]

      # update split state to global state
      m0 = (C_s == 0)
      m1 = (C_s == 1)
      C_s[m0] = c_i
      C_s[m1] = K
      C[s_m] = C_s

      # update cached values and append to end for new cluster
      M[c_i,:] = M_s[0,:]
      MM[c_i,:] = MM_s[0,:]
      G[c_i,:,:] = G_s[0,:,:]
      S[c_i,:] = S_s[0,:]

      M  = np.vstack([M, np.zeros(N_)])
      MM = np.vstack([MM, np.zeros(N_)])
      G_p = G
      G = np.zeros((K+1,N_,3))
      G[:K,:,:] = G_p
      S = np.vstack([S, np.zeros(1)])

      M[K,:] = M_s[1,:]
      MM[K,:] = MM_s[1,:]
      G[K,:,:] = G_s[1,:,:]
      S[K,:] = S_s[1,:]
      K += 1

      print 'split taken!', acc
      print 'splitting cluster', C[i]
      print '  - (old, new) = {}, {}'.format(np.sum(m0), np.sum(m1))
      labels = set(map(lambda(r): true_labels_map[r], sel_rids))
      bad_split = (len(labels) == 1)

      #if min(n_ci, n_cj) == 1:
      #  print 'wtf bad split'
      #  print 
      #  die
      if bad_split and acc  == 1:
        print 'split a pure cluster!!'
        print 'acc_log', acc_log_prior + acc_log_ll + acc_log_trans
        print 'acc', np.exp(acc_log_prior + acc_log_ll + acc_log_trans)
        print
        print 'acc_log_prior', acc_log_prior
        print 'acc_log_ll   ', acc_log_ll
        print 'logPs_split  ', logPs_split 
        print 'logPs_merge  ', logPs_merge
        print 'acc_log_trans', acc_log_trans
        print 

        print 'merge sum M', np.sum(t_M[0,:])
        print 'merge sum MM', np.sum(t_MM[0,:])

        print 'top 10 mismatch positions'
        mm_pos = np.argsort(t_MM[0,:])[::-1][:10]
        for j in mm_pos:
          print 'snp idx', j
          print '  snp', snps_map[j]
          print '  M, MM', t_M[0,j], t_MM[0,j]

        print 'split c0 sum M', np.sum(M[c_i,:])
        print 'split c0 sum MM', np.sum(MM[c_i,:])
        print 'split c1 sum M', np.sum(M[K,:])
        print 'split c1 sum MM', np.sum(MM[K,:])
        die

    else:
      pass 
      #print 'not splitting'
      #cnt = Counter(map(lambda(r): true_labels_map[r], sel_rids))
      #mix_cluster = len(cnt) == 2 and cnt.most_common()[1][0] > 100
      #c0_labels = []
      #c1_labels = []
      #if legit_split and mix_cluster:
      #  for rid, k_s in zip(sel_rids, range(num_s)):
      #    label = true_labels_map[rid]
      #    if C_s[k_s] == 0:
      #      c0_labels.append(label)
      #    else:
      #      c1_labels.append(label)
      #  print 'missed chance to split mixed cluster!', c_i
      #  print '  - proposed split'
      #  print '    - c0', Counter(c0_labels).most_common()
      #  print '    - c1', Counter(c1_labels).most_common()
      #  print 'acc_log', acc_log_prior + acc_log_ll + acc_log_trans
      #  print 'acc', np.exp(acc_log_prior + acc_log_ll + acc_log_trans)
      #  print
      #  print 'acc_log_prior', acc_log_prior
      #  print 'acc_log_ll   ', acc_log_ll
      #  print 'logPs_split  ', logPs_split 
      #  print 'logPs_merge  ', logPs_merge
      #  print 'acc_log_trans', acc_log_trans
      #  print 
      #  print 'merge sum M', np.sum(t_M[0,:])
      #  print 'merge sum MM', np.sum(t_MM[0,:])
      #  print
      #  print 'split c0 sum M', np.sum(M_s[0,:])
      #  print 'split c0 sum MM', np.sum(MM_s[0,:])
      #  print 'split c1 sum M', np.sum(M_s[1,:])
      #  print 'split c1 sum MM', np.sum(MM_s[1,:])
      #  die

  # merge operation
  else:
    #print 'merge operation'
    labels = set(map(lambda(r): true_labels_map[r], sel_rids))
    good_merge = (len(labels) == 1)
    # transition from launch to original split state
    #print 'C_launch', C_launch
    C_s, trans_logP = gibbs_scan(
      A_s, 2, C_launch, M_s, MM_s, G_s, S_s,
      trans_path=C_orig,
      fixed_rids=fixed_rids,
    )
    #print 'C_orig', C_orig
    #print 'C_launch', C_launch
    #print 'trans_logP', trans_logP
    #die

    logPs_split, _, _, _, _ = score(A_s, 2, C_orig)
    n_ci = np.sum(C_orig == 0)
    n_cj = np.sum(C_orig == 1)
    acc_log_prior = (
      gammaln(n_ci + n_cj)  -
      (gammaln(n_ci) + gammaln(n_cj)) - 
      np.log(DP_alpha)
    )
    acc_log_ll = np.sum(logPs_merge) - np.sum(logPs_split)
    acc_log_trans = trans_logP

    acc = min(1, np.exp(acc_log_prior + acc_log_ll + acc_log_trans))

    if random.random() <= acc:
      print 'merge taken! between {} and {}'.format(c_i, c_j)
      taken = True
      # update merge state in global state
      C[s_m] = c_i
      m = np.ones(K)
      m[c_j] = 0
      m = np.ma.make_mask(m)

      # update cached values for new cluster
      M[c_i,:] = t_M[0,:]
      MM[c_i,:] = t_MM[0,:]
      G[c_i,:,:] = t_G[0,:,:]
      S[c_i,:] = t_S[0,:]

      # drop old haplotype cluster c_j
      M = M[m,:]
      MM = MM[m,:]
      G = G[m,:,:]
      S = S[m,:]
      # shift index assignments down
      C[C > c_j] -= 1
      K -= 1

    else:
      pass
      if good_merge:
        print 'rejected good merge between clusters {}, {}'.format(
          c_i, c_j)
        print '  - with {} and {} reads'.format(n_ci, n_cj)
        print 'acc_log_prior', acc_log_prior
        print 'logPs_merge', logPs_merge
        print 'logPs_split', logPs_split
        print 'acc_log_ll', acc_log_ll
        print 'trans_logP', trans_logP
        print 'C_orig', C_orig
        print 'C_launch', C_launch
        print 
        print acc_log_prior + acc_log_ll + acc_log_trans
        die

  return taken, K, C, M, MM, G, S

#--------------------------------------------------------------------------
# gibbs scan
#--------------------------------------------------------------------------
def gibbs_scan(A, K, C, M, MM, G, S, trans_path=None, fixed_rids=None):
  trans_logP = 0.
  M_, N_ = A.shape

  if trans_path is not None:
    assert trans_path.shape[0] == M_, \
      "transition path not fully specified, wrong dimension"
  #print 'gibbs scan'
  #print 'C', C
  for i_p in xrange(M_):
  
    if fixed_rids and i_p in fixed_rids:
      continue

    r_i = A[i_p,:]
    k_p = C[i_p]
    k_fixed = None if trans_path is None else trans_path[i_p]
  
    # matches with current assignment
    m, mm = get_ref_alt(r_i)
    
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
  
    # score read under all haps
    # FIXME add possiblity for read to open a new cluster
    scores = np.ones(K)
    for k in xrange(K):
      n_k = np.sum(C == k)
      if n_k == 0:
        n = np.sum(m) + np.sum(mm)
        scores[k] = np.log(DP_alpha) + n*(np.log(alpha) - np.log(alpha+beta))
        n_k = DP_alpha
      elif k == k_p:
        scores[k] = np.log(n_k) + score_read(m, mm, M_p, MM_p)
      else:
        #assert n_k>0, "i_p {}".format(i_p)
        scores[k] = np.log(n_k) + score_read(m, mm, M[k,:], MM[k,:])
  
    # pick new cluster with prob proportional to scores under K
    # different betas
    log_scores = scores - logsumexp(scores)
    scores = np.exp(log_scores)
    #print 'i_p', i_p
    #print 'scores', scores

    assn = np.random.multinomial(1, scores)
    # take fixed transition path if specified
    if k_fixed is not None:
      k_n = k_fixed
    else:
      k_n = np.nonzero(assn == 1)[0][0]
    trans_logP += log_scores[k_n]
  
    # update haplotypes with new assignment
    if k_n != k_p:
      # update previous haplotype to remove r_i
      M[k_p,:] = M_p
      MM[k_p,:] = MM_p
      G[k_p,:,:] = G_p
      C[i_p] = k_n
      # update next haplotype to add r_i
      M[k_n,:]  = M[k_n,:] + m
      MM[k_n,:] = MM[k_n,:] + mm
      for j in np.nditer(np.nonzero(A[i_p,:] != 0)):
        G[k_n,j,:] = score_beta_site(M[k_n,j], MM[k_n,j])
    else:
      C[i_p] = k_p

  return C, trans_logP

#--------------------------------------------------------------------------
# phasing
#--------------------------------------------------------------------------
def phase(scratch_path):
  global G_seed

  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A, true_labels = util.load_phase_inputs(h5_path)

  print 'loaded {} X {}'.format(len(bcodes), len(snps))
  bcodes, A, true_labels = util.subsample_reads(bcodes, A, true_labels, lim=5000)
  print '  - subsample to {} X {}'.format(len(bcodes), len(snps))

  # FIXME change to assert
  print 'number nonzero', np.sum(np.any((A != 0), axis=1))

  true_labels_map = dict(enumerate(true_labels))
  bcodes_map = dict(enumerate(bcodes))
  snps_map = dict(enumerate(snps))

  M_, N_ = A.shape

  # hidden haplotypes
  K, C = util.get_initial_state(A, hp.K)

  # initialize and save intermediate values for fast vectorized computation
  logP, M, MM, G, S = score(A, K, C)

  # make sure intermediate tables are all consistent
  def assert_state(A, K, C, M, MM, G):
    fail = False
    for k in xrange(K):
      C_i = (C == k)
      assert np.sum(C_i) > 0
      A_k = A[C_i,:]
      for j in xrange(N_):
        M_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == 1, j]))
        MM_c = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -1, j]))
        assert M[k,j]  == M_c
        assert M[k,j]  == M_c
        assert M[k,j]  == M_c
        assert MM[k,j] == MM_c
        assert (G[k,j,:] == score_beta_site(M_c, MM_c)).all()

  # compute seed gammas for empty haplotype cluster
  G_seed = np.zeros((N_, 3))
  G_seed[:,0], G_seed[:,1], G_seed[:,2] = \
    score_beta_site(np.zeros(N_), np.zeros(N_))

  assert_state(A, K, C, M, MM, G)

  assert sample_step * num_samples < num_iterations, \
    "{} iterations is not enough for {} samples with step {}".format(
      num_iterations,
      num_samples,
      sample_step,
    )

  sidx = 0
  C_samples = []
  #H_samples = np.zeros((num_iterations, K, N_))
  #C_samples = np.zeros((num_iterations, M_))
  print 'initial  accuracies', score_assignment(K, C, true_labels_map)
  for iteration in xrange(num_iterations):

    #print 'iteration', iteration
    if iteration % 50 == 0:
      print 'iteration', iteration
      print '  - accuracies',
      print score_assignment(K, C, true_labels_map)

    assert_state(A, K, C, M, MM, G)
    taken, K, C, M, MM, G, S = merge_split(A, K, C, M, MM, G, S,
      true_labels_map, bcodes_map, snps_map)
    assert_state(A, K, C, M, MM, G)
    if taken:
      print 'finished on taken move!'
      print '  - pre gibbs accuracies' 
      print score_assignment(K, C, true_labels_map)
      ##sys.exit(0)
      for i in xrange(3):
        C, _ = gibbs_scan(A, K, C, M, MM, G, S)
      print '  - post gibbs accuracies' 
      print score_assignment(K, C, true_labels_map)

    assert_state(A, K, C, M, MM, G)

    C_samples.append(C)

    ## save all samples from each iteration for now
    #C_samples[sidx,:] = C
    #sidx = (sidx + 1) % num_iterations

  assert_state(A, K, C, M, MM, G)

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

