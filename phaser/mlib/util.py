import os
import pysam
import vcf
from collections import defaultdict, Counter
import numpy as np
import h5py
import cPickle as pickle

#--------------------------------------------------------------------------
# sys
#--------------------------------------------------------------------------

def mkdir_p(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def write_pickle(path, obj):
  f = open(path,'w')
  pickle.dump(
    obj,
    f,  
    pickle.HIGHEST_PROTOCOL
  )
  f.close()

def load_pickle(path):
  f = open(path,'r')
  obj = pickle.load(f)
  f.close()
  return obj 

#--------------------------------------------------------------------------
# 10x
#--------------------------------------------------------------------------

def get_label(read):
  filt_list = filter(lambda(k, v): k == 'AB', read.tags)
  assert len(filt_list) <= 1
  if filt_list == []: 
    return None
  else:
    label = filt_list[0][1].strip()
    return label

def get_barcode(read):
  filt_list = filter(lambda(k, v): k in ['BC', 'BX'], read.tags)
  assert len(filt_list) <= 1
  if filt_list == []: 
    return None
  else:
    barcodepre = filt_list[0][1].strip()
    return barcodepre

#--------------------------------------------------------------------------
# vcf processing
#--------------------------------------------------------------------------

def get_variants(vcf_path):
  vcf_fin = vcf.Reader(filename=vcf_path)
  var_map = {}

  for record in vcf_fin:
    if (
      not record.is_snp or 
      not len(record.ALT) == 1
    ):
      continue
    ref = str(record.REF).lower()
    alt = str(record.ALT[0].sequence).lower()
    pos = record.POS - 1
    var_map[(record.CHROM, pos)] = (ref, alt)
  return var_map

#--------------------------------------------------------------------------
# build/load input read X barcode matrix
#--------------------------------------------------------------------------

def load_phase_inputs(h5_path):
  h5f = h5py.File(h5_path, 'r')
  snps = map(lambda(k,v): (str(k),int(v)), h5f['snps'])
  bcodes = map(lambda(k): str(k), h5f['bcodes'])
  true_labels = map(lambda(k): str(k), h5f['true_labels'])
  A = np.array(h5f['genotypes'])
  h5f.close()
  A = get_normalized_genotypes(A)
  return snps, bcodes, A, true_labels

def make_inputs(bam_path, vcf_path, scratch_path, bcodes=None):

  var_map = get_variants(vcf_path)

  bam_fin = pysam.Samfile(bam_path, 'rb')
  snp_bcode_counts_map = defaultdict(lambda:defaultdict(Counter))
  seen_set = set()
  bcode_label_map = defaultdict(lambda:None)
  for (i, target_snp) in enumerate(sorted(var_map)):
    (ctg, pos) = target_snp
    #if ctg in ['contig-100_6', 'contig-100_10']:
    #  print 'skipping bad ctg', ctg
    #  continue
    if i % 500 == 0:
      print 'num snps', i
      #if i > 400:
      #  break
      #break
    bbegin = pos
    eend = pos + 1
    for pcol in bam_fin.pileup(ctg, bbegin, eend, max_depth=999999999):
      snp = (ctg, pcol.pos)
      if snp != target_snp:
        continue
      assert snp not in seen_set
      seen_set.add(snp)
      #print 'snp {}:{} depth {}'.format(ctg, pos, pcol.n)
      for read in pcol.pileups:
        bcode = get_barcode(read.alignment)
        label = get_label(read.alignment)
        if label:
          bcode_label_map[bcode] = label
        if bcode == None:
          continue
        # skip gapped alignment positions
        if read.is_del or read.is_refskip:
          continue
        snp = (ctg, pcol.pos)
        allele = str(read.alignment.query_sequence[read.query_position]).lower()
        snp_bcode_counts_map[snp][bcode][allele] += 1

  bam_fin.close()

  # filter snps based on within barcode conflicts + coverage
  vcf_fin = vcf.Reader(filename=vcf_path)
  pass_vcf_path = os.path.join(scratch_path, 'pass.vcf')
  filt_vcf_path = os.path.join(scratch_path, 'filt.vcf')
  pass_vcf_fout_ = open(pass_vcf_path, 'wb')
  filt_vcf_fout_ = open(filt_vcf_path, 'wb')
  pass_vcf_fout = vcf.Writer(pass_vcf_fout_, vcf_fin)
  filt_vcf_fout = vcf.Writer(filt_vcf_fout_, vcf_fin)

  def is_mix(c1, c2):
    if c2 > c1:
      c1, c2 = c2, c1
    ## pass
    #return c2 < 0.05 * c1 or (c2 == 1 and c1 > 5)
    # mix
    return c2 > 0.05 * c1 and not (c2 == 1 and c1 > 5)

  pass_snps = []
  for record in vcf_fin:
    snp = (record.CHROM, record.POS - 1)
    if snp not in var_map:
      continue
    ref_allele, alt_allele = var_map[snp]
    ref_calls = 0
    ref_mix   = 0
    alt_calls = 0
    alt_mix   = 0
    for bcode, counts in snp_bcode_counts_map[snp].items():
      rc, ac = (counts[ref_allele], counts[alt_allele])
      if rc > 0 and rc >= ac:
        ref_calls += 1
        ref_mix += is_mix(rc, ac)
      elif ac > 0 and ac > rc:
        alt_calls += 1
        alt_mix += is_mix(rc, ac)
   
    record.INFO = {}
    record.INFO['filters'] = 'ref:{};ref-m:{};alt:{};alt-m:{};'.format(
      ref_calls, ref_mix,
      alt_calls, alt_mix,
    )
    if (
      ref_calls >= 10 and alt_calls >= 10 and
      ref_mix <= 0.05 * ref_calls and
      alt_mix <= 0.05 * alt_calls
    ):
      pass_vcf_fout.write_record(record)
      pass_snps.append(snp)
    else:
      filt_vcf_fout.write_record(record)

  # filter for barcodes covering >1 informative snp
  # remove mixed calls from barcodes
  pass_snps_set = set(pass_snps)
  bcode_snp_counts = Counter()
  for snp, bcode_counts_map in snp_bcode_counts_map.items():
    if snp not in pass_snps_set:
      continue
    for bcode, counts in bcode_counts_map.items():
      ra, aa = var_map[snp]
      rc, ac = counts[ra], counts[aa]
      if not is_mix(rc, ac) and (rc > 0 or ac > 0):
        bcode_snp_counts[bcode] += 1

  true_labels = []
  pass_bcodes = []
  for bcode, cnt in bcode_snp_counts.most_common():
    # skip if barcode not in subset specified
    if bcodes and bcode not in bcodes:
      continue
    if cnt > 1:
      pass_bcodes.append(bcode)
      if bcode_label_map[bcode]:
        true_labels.append(bcode_label_map[bcode])
      else:
        true_labels.append(0)
  pass_bcodes_set = set(pass_bcodes)

  # create input matrix
  idx_snp_map = dict(list(enumerate(pass_snps)))
  idx_rid_map = dict(list(enumerate(pass_bcodes)))
  snp_idx_map = dict(map(lambda(k,v): (v,k), idx_snp_map.items()))
  rid_idx_map = dict(map(lambda(k,v): (v,k), idx_rid_map.items()))
  M = len(rid_idx_map)
  N = len(snp_idx_map)
  A = np.zeros((M, N))
  for j, snp in idx_snp_map.items():
    ref_allele, alt_allele = var_map[snp]
    for bcode, counts in snp_bcode_counts_map[snp].items():
      if bcode not in pass_bcodes_set:
        continue
      i = rid_idx_map[bcode]
      rc, ac = (counts[ref_allele], counts[alt_allele])
      # skip mixed
      if is_mix(rc, ac):
        continue
      if rc > ac:
        A[i,j] = -rc
      elif ac > rc:
        A[i,j] = ac

  pass_snps_vec = np.array(pass_snps)
  pass_bcodes_vec = np.array(pass_bcodes)

  print '{} bcodes X {} snps'.format(M, N)
  print 'total genotype entries {}'.format(np.sum(np.abs(A)))

  h5_path = os.path.join(scratch_path, 'inputs.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('snps', data=pass_snps)
  h5f.create_dataset('bcodes', data=pass_bcodes)
  h5f.create_dataset('genotypes', data=A)
  h5f.create_dataset('true_labels', data=true_labels)
  h5f.close()

def make_inputs_toy(d1, d2, scratch_path):
#def make_inputs_toy(d1, d2, scratch_path):
  # toy example
  pass_snps = [
    ('snp0',0),
    ('snp1',0),
    ('snp2',0),
    ('snp3',0),
    ('snp4',0),
  ]
  #pass_bcodes = [
  #  'bcode0',
  #  'bcode1',
  #  'bcode2',
  #  'bcode3',
  #  'bcode4',
  #]
  #A = np.array([
  #[-1,1,1,1,1],
  #[1,-1,1,1,1],
  #[1,1,-1,1,1],
  #[1,1,1,-1,1],
  #[1,1,1,1,-1],
  #])
  #true_labels = [
  #  'c0',
  #  'c1',
  #  'c2',
  #  'c3',
  #  'c4',
  #]
  pass_bcodes = [
    'bcode0',
    'bcode1',
    'bcode2',
    'bcode3',
    'bcode4',
    'bcode5',
    'bcode6',
    'bcode7',
    'bcode8',
    'bcode9',

    'bcode15',
    'bcode16',
    'bcode17',
    'bcode18',
    'bcode19',
    'bcode25',
    'bcode26',
    'bcode27',
    'bcode28',
    'bcode29',
  ]
  A = np.array([
  [-1,1,1,1,1],
  [1,-1,1,1,1],
  [1,1,-1,1,1],
  [1,1,1,-1,1],
  [1,1,1,1,-1],
  [-1,1,1,1,1],
  [1,-1,1,1,1],
  [1,1,-1,1,1],
  [1,1,1,-1,1],
  [1,1,1,1,-1],

  [-1,1,1,1,1],
  [1,-1,1,1,1],
  [1,1,-1,1,1],
  [1,1,1,-1,1],
  [1,1,1,1,-1],
  [-1,1,1,1,1],
  [1,-1,1,1,1],
  [1,1,-1,1,1],
  [1,1,1,-1,1],
  [1,1,1,1,-1],
  ])
  true_labels = [
    'c0',
    'c1',
    'c2',
    'c3',
    'c4',
    'c0',
    'c1',
    'c2',
    'c3',
    'c4',

    'c0',
    'c1',
    'c2',
    'c3',
    'c4',
    'c0',
    'c1',
    'c2',
    'c3',
    'c4',
  ]

  h5_path = os.path.join(scratch_path, 'inputs.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('snps', data=pass_snps)
  h5f.create_dataset('bcodes', data=pass_bcodes)
  h5f.create_dataset('genotypes', data=A)
  h5f.create_dataset('true_labels', data=true_labels)
  h5f.close()

def make_inputs_toy2(d1, d2, scratch_path):
  # toy example
  pass_snps = [
    ('snp0',0),
    ('snp1',0),
    ('snp2',0),
    ('snp3',0),
    ('snp4',0),
  ]
  pass_bcodes = [
    'bcode0',
    'bcode1',
    'bcode2',
    'bcode3',
    'bcode4',
    'bcode5',
    'bcode6',
    'bcode7',
    'bcode8',
    'bcode9',
  ]
  A = np.array([
  [-1,1,1,1,1],
  [-1,1,1,1,1],
  [-1,1,1,1,1],
  [-1,1,1,1,1],
  [-1,1,1,1,1],

  [1,-1,1,1,1],
  [1,-1,1,1,1],
  [1,-1,1,1,1],
  [1,-1,1,1,1],
  [1,-1,1,1,1],
  ])
  true_labels = [
    'c0',
    'c0',
    'c0',
    'c0',
    'c0',
    'c1',
    'c1',
    'c1',
    'c1',
    'c1',
  ]

  h5_path = os.path.join(scratch_path, 'inputs.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('snps', data=pass_snps)
  h5f.create_dataset('bcodes', data=pass_bcodes)
  h5f.create_dataset('genotypes', data=A)
  h5f.create_dataset('true_labels', data=true_labels)
  h5f.close()

#--------------------------------------------------------------------------
# phasing utilities
#--------------------------------------------------------------------------
def make_bin_outputs(
  clusters_map,
  inbam_path,
  outdir_path,
):
  bcode_cidx_map = defaultdict(lambda:None)
  for cidx, bcodes in clusters_map.items():
    for bcode in bcodes:
      bcode_cidx_map[bcode] = cidx

  # save raw clusters
  out_path = os.path.join(outdir_path, 'clusters.p')
  write_pickle(out_path, clusters_map)

  # text file of barcodes in each cluster
  for k, bcodes in clusters_map.items():
    out_path = os.path.join(outdir_path, '{}.bin.txt'.format(k)) 
    with open(out_path, 'w') as fout:
      for bcode in bcodes:
        fout.write('{}\n'.format(bcode))

  # dump output bams
  #return
  inbam = pysam.Samfile(inbam_path, 'rb')

  unassn_path = os.path.join(outdir_path, 'unassigned.bam')
  bam_fouts = {None:pysam.Samfile(unassn_path, 'wb', template=inbam)}
  for cidx in clusters_map:
    outbam_path = os.path.join(outdir_path, 'cluster.{}.bam'.format(cidx))
    bam_fouts[cidx] = pysam.Samfile(outbam_path, 'wb', template=inbam)
  
  for read in inbam:
    bcode = get_barcode(read)
    cidx = bcode_cidx_map[bcode]
    bam_fouts[cidx].write(read)

  inbam.close()
  for fout in bam_fouts.values():
    fout.close()

def get_initial_state(A, true_labels_map=None):
  M_, N_ = A.shape
  # initialize in a single cluster
  K = 1
  C = np.zeros(M_, dtype=np.int)

  #if true_labels_map:
  #  label_cid_map = dict(map(
  #    lambda(cid, l): (l, cid),
  #    enumerate(set(true_labels_map.values())),
  #  ))
  #  K = len(label_cid_map)
  #  for rid, label in sorted(true_labels_map.items()):
  #    C[rid] = label_cid_map[label]

  ## initialize with every read in its own cluster
  #K = M_
  #C = np.arange(M_, dtype=np.int)
  return K, C

def get_initial_state_fixedK(A, K):
  # get inspired by some reads to initialize the hidden haplotypes
  # FIXME not sure of implication if 0s persist beyond init....
  M_, N_ = A.shape
  #H = np.ones((K,N_))
  H = np.zeros((K,N_))
  C = np.random.choice(K, M_)
  # pass through reads and greedily assign to cluster with the fewest
  # mismatches per assignment
  for i in xrange(M_):
    r = A[i,:]
    mms = np.zeros(K)
    for k in xrange(K):
      mms[k] = np.sum(np.abs((H[k,:] * np.sign(r)).clip(max=0)))
    k_c = np.argmin(mms)
    # overwrite current haplotype value with this read's nonzero calls
    H[k_c,(r != 0)] = np.sign(r[r != 0])
    C[i] = k_c

  # every cluster must have at least one read for now
  assert M_ >= K, "initial K={} too high, more than {} reads".format(K, M_)
  for k in xrange(K):
    if not (C == k).any():
      C[k] = k 
    print '{} reads assigned to hap {}'.format(
      np.sum(C == k),
      k
    )
  print 'init H'
  print H
  print 'init C'
  print C

  return H, C

def get_normalized_genotypes(_A):
  A = np.array(_A)
  A[A>0] = 1
  A[A<0] = -1
  return A

def subsample_reads(bcodes, A, true_labels, lim=15000):
  # subsample most informative barcodes
  A_z = np.sum((A == 0), axis=1)
  s_idx = np.argsort(A_z)
  #for z, cnt in sorted(Counter(A_z).items(), reverse=True):
  #  print len(snps) - z, cnt
  A = A[s_idx,:][:lim,:]

  last = min(lim, A.shape[0])-1
  print 'min occupancy', A.shape[1] - A_z[s_idx][last]
  bcodes = map(str, np.array(bcodes)[s_idx][:lim])
  true_labels = map(str, np.array(true_labels)[s_idx][:lim])
  return bcodes, A, true_labels


