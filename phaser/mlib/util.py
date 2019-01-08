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

def get_barcode_rc(read):
  filt_list = filter(lambda(k, v): k in ['BC', 'BX'], read.tags)
  assert len(filt_list) <= 1
  if filt_list == []: 
    return None
  else:
    barcodepre = filt_list[0][1].strip()
    return barcodepre

def get_barcode_sms(read):
  return read.qname

get_barcode = get_barcode_rc

def set_sms(sms_on):
  global get_barcode
  get_barcode = get_barcode_sms if sms_on else get_barcode_rc

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
MIN_ALLELE_COUNTS = 5

def load_phase_inputs(h5_path):
  h5f = h5py.File(h5_path, 'r')
  snps = map(lambda(k,v): (str(k),int(v)), h5f['snps'])
  bcodes = map(lambda(k): str(k), h5f['bcodes'])
  true_labels = map(lambda(k): str(k), h5f['true_labels'])
  A = np.array(h5f['genotypes'])
  h5f.close()
  #A = get_normalized_genotypes(A)
  return snps, bcodes, A, true_labels

def make_inputs(bam_path, vcf_path, scratch_path, bcodes=None):

  var_map = get_variants(vcf_path)

  bam_fin = pysam.Samfile(bam_path, 'rb')
  snp_bcode_counts_map = defaultdict(lambda:defaultdict(Counter))
  seen_set = set()
  bcode_label_map = defaultdict(lambda:0)
  all_bcodes = set()
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
        all_bcodes.add(bcode)
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
      ref_calls >= MIN_ALLELE_COUNTS and 
      alt_calls >= MIN_ALLELE_COUNTS and
      ref_mix <= 0.05 * ref_calls and
      alt_mix <= 0.05 * alt_calls
    ):
      pass_vcf_fout.write_record(record)
      pass_snps.append(snp)
    else:
      filt_vcf_fout.write_record(record)

  pass_bcodes = list(all_bcodes)
  true_labels = map(lambda(b): bcode_label_map[b], pass_bcodes)

  # create input matrix
  idx_snp_map = dict(list(enumerate(pass_snps)))
  idx_rid_map = dict(list(enumerate(all_bcodes)))
  snp_idx_map = dict(map(lambda(k,v): (v,k), idx_snp_map.items()))
  rid_idx_map = dict(map(lambda(k,v): (v,k), idx_rid_map.items()))
  M = len(rid_idx_map)
  N = len(snp_idx_map)
  A = np.zeros((M, N))
  for j, snp in idx_snp_map.items():
    ref_allele, alt_allele = var_map[snp]
    for bcode, counts in snp_bcode_counts_map[snp].items():
      #if bcode not in pass_bcodes_set:
      #  continue
      i = rid_idx_map[bcode]
      rc, ac = (counts[ref_allele], counts[alt_allele])
      # skip mixed
      if is_mix(rc, ac):
        continue
      if rc > ac:
        A[i,j] = -rc
      elif ac > rc:
        A[i,j] = ac

  A = get_normalized_genotypes(A)
  # snps must have at least four supporting barcodes
  mask = (np.sum(np.abs(A), axis=0) > 3)
  ## FIXME tmp
  #mask = (np.sum(np.abs(A), axis=0) > 1)
  A = A[:,mask]
  pass_snps = list(np.array(pass_snps)[mask])
  print '  - dropped {} snps with < 4 supporting barcodes'.format(np.sum(~mask))

  # include only reads with at least two snps
  mask = (np.sum(np.abs(A), axis=1) > 1)
  A = A[mask,:]
  pass_bcodes = list(np.array(pass_bcodes)[mask])
  true_labels = list(np.array(true_labels)[mask])
  print '  - dropped {} barcodes not overlapping >= 2 hets'.format(np.sum(~mask))

  # snps must have ref and alt barcode calls
  mref = (np.sum((A == -1), axis=0) > 0)
  malt = (np.sum((A ==  1), axis=0) > 0)
  mask = mref & malt
  A = A[:,mask]
  pass_snps = list(np.array(pass_snps)[mask])
  print '  - dropped {} snps without het barcode calls'.format(np.sum(~mask))

  print '{} bcodes X {} snps'.format(M, N)
  print 'total genotype entries {}'.format(np.sum(np.abs(A)))

  h5_path = os.path.join(scratch_path, 'inputs.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('snps', data=pass_snps)
  h5f.create_dataset('bcodes', data=pass_bcodes)
  h5f.create_dataset('genotypes', data=A)
  h5f.create_dataset('true_labels', data=true_labels)
  h5f.close()

def make_inputs_toy(
  M = 400,
  N = 1000,
  #N = 2000,
  #mm = 1000,
  mm = 20,
  sparsity = 0,
  #err = 0.00,
  err = 0.01,
):

  assert mm < N
  assert sparsity < 1 and sparsity >= 0

  mid = int(M/2)

  A_1 = np.zeros(N)
  A_2 = np.array(A_1)
  A_2[0:mm] = 1

  # draw reads
  A = np.random.random_sample((M, N))
  mask = (A < err)
  A[mask]  = 1
  A[~mask] = -1

  # flip bottom half of reads that are supposed to match the alternate
  # allele
  fmask = (A[mid:,:mm] == 1)
  A[mid:,:mm][fmask]  = -1
  A[mid:,:mm][~fmask] = 1

  # impose sparsity
  num_mask = int(sparsity * A.size)
  smask = np.full(A.size, False)
  smask[:num_mask] = True
  np.random.shuffle(smask)
  smask = smask.reshape((M,N))

  pass_snps = map(lambda(i): ('ctg', i), xrange(N))
  pass_bcodes = map(lambda(i): 'bcode{}'.format(i), xrange(M))
  true_labels = ['c0'] * mid
  true_labels.extend(['c1'] * (M - mid))
  
  return A, pass_snps, pass_bcodes, true_labels,


def make_inputs_toy2(d1, d2, scratch_path):
  # toy example
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
  M, N = A.shape
  pass_snps = map(lambda(i): ('ctg', i), xrange(N))
  pass_bcodes = map(lambda(i): 'bcode{}'.format(i), xrange(M))
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

def make_inputs_toy3(d1, d2, scratch_path):
  # toy example
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
  M, N = A.shape
  pass_snps = map(lambda(i): ('ctg', i), xrange(N))
  pass_bcodes = map(lambda(i): 'bcode{}'.format(i), xrange(M))
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
  A = A[s_idx,:][:lim,:]

  last = min(lim, A.shape[0])-1
  print 'min occupancy', A.shape[1] - sum((A[last,:] == 0))
  bcodes = map(str, np.array(bcodes)[s_idx][:lim])
  true_labels = map(str, np.array(true_labels)[s_idx][:lim])
  return bcodes, A, true_labels


