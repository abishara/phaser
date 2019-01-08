import os
import sys
from scipy.special import gammaln

import util

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
    #print 'cluster', k 
    #pp(cnt)
    print 'c{}:{} '.format(k, total),
  print
  return sorted(accs)

def score(A, K, C, debug=False):
  M_, N_ = A.shape
  # match+mismatch counts for each site in each hap K
  M  = np.zeros((K, N_))
  MM = np.zeros((K, N_))
  S = np.zeros((K, 1))
  h_p = np.log(0.5)
  h_n = np.log(0.5)
  p_m = np.log(0.99)
  p_mm = np.log(0.01)

  # score assigned reads in each cluser by VE on h
  for k in xrange(K):
    C_i = (C == k)
    A_k = A[C_i,:]
    for j in xrange(N_):
      M[k,j]  = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == 1, j]))
      MM[k,j] = np.sum(np.abs(A_k[np.sign(A_k[:,j]) == -1, j]))
    p = np.vstack([
      h_p + M[k,:] * p_m + MM[k,:] * p_mm,
      h_n + MM[k,:] * p_m + M[k,:] * p_mm
    ])
    S[k] = np.prod(logsumexp(ps, axis=0))

  return M, MM, S

def score_read(m, mm, M, MM):
  logP = 0.

  mask0 = (m > 0)
  mask1 = (mm > 0)
  N = M + MM

  # compute posterior at each site
  p = np.vstack([
    h_p + M[k,:] * p_m + MM[k,:] * p_mm,
    h_n + MM[k,:] * p_m + M[k,:] * p_mm
  ])
  p -= logsumexp(p, axis=0)
  p[0,mask1] * p_m, p[1,mask1] *

  # match component
  mlogP = np.sum(
    (
      np.log(alpha_v[mask0] + M[mask0]) - 
      np.log(alpha_v[mask0] + beta_v[mask0] + N[mask0])
    ) * m[mask0]
  )
  # mismatch component
  mmlogP = np.sum(
    (
      np.log(beta_v[mask1] + MM[mask1]) - 
      np.log(alpha_v[mask1] + beta_v[mask1] + N[mask1])
    ) * mm[mask1]
  )
  logP = mlogP + mmlogP
  return logP

A, snps, bcodes, true_labels = util.make_inputs_toy()

true_labels_map = dict(enumerate(true_labels))
bcodes_map = dict(enumerate(bcodes))
snps_map = dict(enumerate(snps))

M_, N_ = A.shape

K = 2
C = np.zeros(M_,dtype=np.int)
#C[int(M_/2):] = 1
C[:1] = 1

logP, M, MM, G, S = score(A, K, C)

print 'init accuracies', score_assignment(K, C, true_labels_map)
_, K, C, M, MM, G, S = \
  gibbs_scan(A, K, C, M, MM, G, S, fixed_rids=[0, 399], l=true_labels_map)
