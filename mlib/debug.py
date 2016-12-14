import os
import numpy as np
from collections import Counter
#import pylab
#import matplotlib
#matplotlib.use('Agg')
#matplotlib.rc('xtick', labelsize=11.5) 
#matplotlib.rc('ytick', labelsize=11.5) 
#matplotlib.rc('ytick', labelsize=11.5) 
##font = {
##  'family' : 'serif',
##  'serif'  : ['times new roman'],
##  'size'   : 9,
##}
##matplotlib.rc('font', **font)
#import matplotlib.pyplot as pyplot
#import matplotlib.ticker as ticker
#from matplotlib.ticker import FuncFormatter
#import datetime
#from matplotlib import dates
#from matplotlib import cm

"""
# maintain plot count
plot_count = 0 
def dump(
  outdir_path,
  H_k,
  A_k,
  sel_rids,
  idx_rid_map,
  idx_snp_map,
):

  M_, N_ = A_k.shape

  # reorder sel_rids, A_k in sorted order of fewest leading zeros 

  # count leading zeros
  Z_k = np.apply_along_axis(np.argmax, 1, A_k != 0)
  s_idxs = np.argsort(Z_k)

  sel_rids = sel_rids[s_idxs]
  A_k = A_k[s_idxs,:]

  perfect_snps = np.nonzero(np.apply_along_axis(np.all, 0, (A_k == 0) | (A_k == H_k)))
  perfect_rids = np.nonzero(np.apply_along_axis(np.all, 1, (A_k == 0) | (A_k == H_k)))
  num_mm = np.apply_along_axis(np.sum, 1, (A_k != H_k) & (A_k != 0))
  num_m  = np.apply_along_axis(np.sum, 1, (A_k == H_k) & (A_k != 0))

  s_idxs = np.argsort(-num_mm)
  A_k = A_k[s_idxs,:]
  sel_rids = sel_rids[s_idxs]
  num_mm = num_mm[s_idxs]

  print 'total reads, snps', M_, N_
  print '  - perfect rids', len(perfect_rids[0])
  print '  - perfect snps', len(perfect_snps[0])

  bad_rids = s_idxs[num_mm > 3]
  B_k = A_k[bad_rids,:]
  snp_mm = np.apply_along_axis(np.sum, 0, (B_k != 0) & (B_k != H_k))
  print 'number of read mismatches per snp for bad reads', len(bad_rids)
  #print sorted(Counter(snp_mm).items())
  print Counter(snp_mm).most_common()
  print
  good_rids = s_idxs[num_mm < 3]
  G_k = A_k[good_rids,:]
  snp_m = np.apply_along_axis(np.sum, 0, (G_k != 0) & (G_k == H_k))
  print 'number of read matches per snp for good reads', len(good_rids)
  #print sorted(Counter(snp_m).items())
  print Counter(snp_m).most_common()
  print
  return

  #die

  fig = pyplot.figure()
  fig.subplots_adjust(bottom=0.4)
  sub = fig.add_subplot(111)
  pyplot.title('hap debug')
  pyplot.xlabel('snps')
  pyplot.ylabel('reads')
    
  # plot ref and alt calls for all barcodes
  #lim = 100
  #lim2 = 500
  #A_k = A_k[:lim,500:600]
  #H_k = H_k[500:600]
  for sgn in [1, -1]:
    ys, xs = np.nonzero(H_k == sgn * A_k)
    sub.plot(
      xs, 
      ys, 
      'bs' if sgn == 1 else 'rs',
    )   

  # plot read lines
  # FIXME TODO

  sub.spines['top'].set_visible(False)
  sub.spines['right'].set_visible(False)
  sub.get_xaxis().tick_bottom()
  sub.get_yaxis().tick_left()


  path = os.path.join(outdir_path, 'reads.png')
  pyplot.savefig(path, dpi=200)
  # NOTE need these to free up memory
  fig.clf()
  #fig.close()
  pyplot.clf()
  pyplot.close()

  return
"""
 
def compute_entropy(
  A_,
):
  # read coverage
  #m  = np.apply_along_axis(np.sum, 0, A.clip(min=0))
  #mm = np.apply_along_axis(np.sum, 0, np.abs(A.clip(max=0)))
  #n  = mm + m

  # barcode coverage
  A = np.array(A_)
  A[A>0] = 1
  A[A<0] = -1
  m  = np.apply_along_axis(np.sum, 0, A.clip(min=0))
  mm = np.apply_along_axis(np.sum, 0, np.abs(A.clip(max=0)))
  n  = mm + m

  mask = (n >= 10)

  A = A[:,mask]
  m = m[mask]
  mm = mm[mask]
  n  = n[mask]
  sel_snps = np.nonzero(mask)[0]

  P = np.vstack([m/n, mm/n])
  entr = np.sum(-np.log2(P) * P, axis=0)

  # set zero entropy sites
  entr[np.isnan(entr)] = 0.

  return n, mask, sel_snps, entr


def plot_entropy(
  k,
  outdir_path,
  A,
  debug=True,
):
  n, mask, sel_snps, entr = compute_entropy(A)

  # for zero entropy sites, go through highest coverage and find one that is 
  nhmask = (entr == 0)
  nhsel_snps = sel_snps[nhmask]
  s_idx = np.argsort(-n[nhmask])

  fp_snps_list = []
  if debug:
    if nhmask.any():
      for j in np.nditer(nhsel_snps[s_idx]):
        rmask = (A[:,j] != 0)
        A_sub = A[rmask,:]
        sub_n, sub_mask, sub_sel_snps, sub_entr = compute_entropy(A_sub)
        if (sub_entr > 0.8).any():
          fp_snps_list.append(
            (int(j), sub_mask, sub_entr)
          )
          #break
  return mask, entr, fp_snps_list

