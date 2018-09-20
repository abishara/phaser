import os
import numpy as np
from scipy.special import gammaln
from collections import Counter

def score_beta_site(m, mm, alpha, beta):
  a = gammaln(m + alpha)
  b = gammaln(mm + beta)
  n = gammaln(m + mm + alpha + beta)
  reg = gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta)
  return a + b - n + reg

num_mm_sites = 32
num_match_sites = 2000
num_match_sites = 1157

scale = 0.05
aa, bb = np.array([0.9, 0.1]) * scale
a, b   = np.array([0.8, 0.2]) * scale
#a, b = 0.05, 0.05
d = 20
d = 15
d = 10
split_cost = (
  2 * num_match_sites * score_beta_site(d, 0, a, b) + 
  num_mm_sites * score_beta_site(d, 0, a, b)
)
merge_cost = (
  num_match_sites * score_beta_site(2*d, 0, a, b) + 
  num_mm_sites * score_beta_site(d, d, a, b)
)

print 'split cost', split_cost
print 'merge cost', merge_cost
print 'delta', split_cost - merge_cost

