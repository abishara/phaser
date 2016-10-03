import os
import sys
import pysam
import tabix
import random
import vcf
from collections import defaultdict, Counter
import math
import numpy as np
from scipy.sparse import lil_matrix
from scipy.misc import logsumexp
import h5py


def mkdir_p(path):
  if not os.path.isdir(path):
    os.makedirs(path)

def get_barcode(read):
  filt_list = filter(lambda(k, v): k == 'BC', read.tags)
  if filt_list == []: 
    return None
  else:
    barcodepre = filt_list[0][1].strip()
    return barcodepre

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

def make_inputs(bam_path, vcf_path, scratch_path):
  
  var_map = get_variants(vcf_path)

  bam_fin = pysam.Samfile(bam_path, 'rb')
  snp_bcode_counts_map = defaultdict(lambda:defaultdict(Counter))
  seen_set = set()
  for (i, (ctg, pos)) in enumerate(sorted(var_map)):
    if i % 10 == 0:
      print 'num snps', i
      #if i > 400:
      #  break
      #break
    bbegin = pos
    eend = pos + 1
    for pcol in bam_fin.pileup(ctg, bbegin, eend):
      snp = (ctg, pcol.pos)
      if snp not in var_map:
        continue
      if snp in seen_set:
        continue
      seen_set.add(snp)
      for read in pcol.pileups:
        bcode = get_barcode(read.alignment)
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

  pass_snps = []
  for record in vcf_fin:
    snp = (record.CHROM, record.POS - 1)
    if snp not in var_map:
      continue
    ref_allele, alt_allele = var_map[snp]
    ref_cnts = 0
    alt_cnts = 0
    mix_bcodes = 0
    for bcode, counts in snp_bcode_counts_map[snp].items():
      ref_cnts += counts[ref_allele] > 0
      alt_cnts += counts[alt_allele] > 0
      mix = (counts[ref_allele] > 0 and counts[alt_allele] > 0)
      mix_bcodes += mix
   
    record.INFO = {}
    record.INFO['filters'] = 'ref:{};alt:{};mix:{};'.format(
      ref_cnts, alt_cnts, mix_bcodes)
    if mix_bcodes < 2 and ref_cnts >= 10 and alt_cnts >= 10:
      pass_vcf_fout.write_record(record)
      pass_snps.append(snp)
    else:
      filt_vcf_fout.write_record(record)

  # filter for barcodes covering at >1 informative snp
  pass_snps_set = set(pass_snps)
  bcode_snp_counts = Counter()
  for snp, bcode_counts_map in snp_bcode_counts_map.items():
    if snp not in pass_snps_set:
      continue
    for bcode in bcode_counts_map:
      bcode_snp_counts[bcode] += 1

  pass_bcodes = []
  for bcode, cnt in bcode_snp_counts.most_common():
    if cnt > 1:
      pass_bcodes.append(bcode)
  pass_bcodes_set = set(pass_bcodes)

  # create input matrix
  idx_snp_map = dict(list(enumerate(pass_snps)))
  idx_rid_map = dict(list(enumerate(pass_bcodes)))
  snp_idx_map = dict(map(lambda(k,v): (v,k), idx_snp_map.items()))
  rid_idx_map = dict(map(lambda(k,v): (v,k), idx_rid_map.items()))
  M = len(rid_idx_map)
  N = len(snp_idx_map)
  #A = lil_matrix((M, N))
  A = np.zeros((M, N))
  for j, snp in idx_snp_map.items():
    ref_allele, alt_allele = var_map[snp]
    for bcode, counts in snp_bcode_counts_map[snp].items():
      if bcode not in pass_bcodes_set:
        continue
      i = rid_idx_map[bcode]
      ref_cnts = counts[ref_allele]
      alt_cnts = counts[alt_allele]
      # skip mixed
      if ref_cnts > 0 and alt_cnts > 0:
        continue
      if ref_cnts > 0:
        A[i,j] = -ref_cnts
      elif alt_cnts > 0:
        A[i,j] = alt_cnts

  pass_snps_vec = np.array(pass_snps)
  pass_bcodes_vec = np.array(pass_bcodes)

  print '{} bcodes X {} snps'.format(M, N)

  # toy example
  #pass_snps = [
  #  ('snp0',0),
  #  ('snp1',0),
  #  ('snp2',0),
  #  ('snp3',0),
  #  ('snp4',0),
  #]
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

  h5_path = os.path.join(scratch_path, 'inputs.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('snps', data=pass_snps)
  h5f.create_dataset('bcodes', data=pass_bcodes)
  h5f.create_dataset('genotypes', data=A)
  h5f.close()

def load_phase_inputs(h5_path):
  h5f = h5py.File(h5_path, 'r')
  snps = map(lambda(k,v): (str(k),int(v)), h5f['snps'])
  bcodes = map(lambda(k): str(k), h5f['bcodes'])
  A = np.array(h5f['genotypes'])
  h5f.close()
  return snps, bcodes, A

def phase(scratch_path):
  h5_path = os.path.join(scratch_path, 'inputs.h5')
  snps, bcodes, A = load_phase_inputs(h5_path)
  idx_snp_map = dict(list(enumerate(snps)))
  idx_rid_map = dict(list(enumerate(bcodes)))
  snp_idx_map = dict(map(lambda(k,v): (v,k), idx_snp_map.items()))
  rid_idx_map = dict(map(lambda(k,v): (v,k), idx_rid_map.items()))

  print 'loaded {} X {}'.format(len(bcodes), len(snps))

  M, N = A.shape
  K = 10
  H = np.ones((K, N))
  prob_map = {
    "logP(h)" : math.log(1. / K),
    "logP(err)"  : math.log(0.01),
    "logP(err')" : math.log(1 - 0.01),
  } 

  S = np.zeros((M, K))
  def score(A, H, i_vec=None, k_vec=None):
    logP = 0.0
    k_iter = xrange(K) if k_vec == None else k_vec
    # FIXME funky hack for nil values of i_vec
    i_iter = xrange(M) if i_vec == None else i_vec
    for i in i_iter:
      #for k in k_iter:
      for k in xrange(K):
        matches = sum(np.abs(A[i,np.sign(A[i,:]) == H[k,:]]))
        mismatches = sum(np.abs(A[i,np.sign(A[i,:]) == -H[k,:]]))
        S[i,k] = ( prob_map["logP(h)"] +
          prob_map["logP(err')"] * matches + 
          prob_map["logP(err)"] * mismatches
        )
      logP += logsumexp(S[i,:])
    return logP

  logP = score(A, H)
  logP_min = logP
  H_min = np.array(H)

  print 'initial logP', logP

  for iteration in xrange(12):
    print 'iteration', iteration
    for k in xrange(K):
      print '  - hap {}, logP: {}'.format(k, logP)
      for j in xrange(N):
        pp = H[k,j]
        sel_rids = list(np.nonzero(A[:,j])[0])
        sel_hap = [k]
        logP_p = score(A, H, sel_rids, sel_hap)
        H[k,j] *= -1
        logP_n = score(A, H, sel_rids, sel_hap)
        delta_logP = -logP_p + logP_n
        P = min(1, np.exp(delta_logP))

        matches = sum(np.abs(A[np.sign(A[:,j]) == H[k,j],j]))
        mismatches = sum(np.abs(A[np.sign(A[:,j]) == -H[k,j],j]))

        #print 'prev matches', len(matches)
        #print 'prev mismatches', len(mismatches)
        #print 'delta log P', delta_logP
        if P > random.random():
          #print '  - accept', delta_logP
          logP += delta_logP
        else:
          #print '  - reject'
          H[k,j] *= -1
        if logP > logP_min:
          logP_min = logP
          H_min = np.array(H)
          #print '**new minimum', logP
        #die
      #die

  #print 'converged', H_min
  h5_path = os.path.join(scratch_path, 'phased.h5')
  h5f = h5py.File(h5_path, 'w')
  h5f.create_dataset('H', data=H_min)
  h5f.close()

def make_outputs(scratch_path):
  inputsh5_path = os.path.join(scratch_path, 'inputs.h5')
  phaseh5_path = os.path.join(scratch_path, 'phased.h5')
  snps, bcodes, A = load_phase_inputs(inputsh5_path)
  h5f = h5py.File(phaseh5_path, 'r')
  H = np.array(h5f['H'])
  h5f.close()
  #print 'loaded', H
  M, N = A.shape
  K = 10
  prob_map = {
    "logP(h)" : math.log(1. / K),
    "logP(err)"  : math.log(0.01),
    "logP(err')" : math.log(1 - 0.01),
  } 
  idx_snp_map = dict(list(enumerate(snps)))
  idx_rid_map = dict(list(enumerate(bcodes)))

  # determine read assignment to haplotypes
  W = np.empty((M,))
  W.fill(-1)
  for i in xrange(M):
    S = np.zeros((K,))
    for k in xrange(K):
      matches = sum(np.abs(A[i,np.sign(A[i,:]) == H[k,:]]))
      mismatches = sum(np.abs(A[i,np.sign(A[i,:]) == -H[k,:]]))
      S[k] = ( prob_map["logP(h)"] +
        prob_map["logP(err')"] * matches + 
        prob_map["logP(err)"] * mismatches
      )
    # renormalize
    S -= logsumexp(S)
    assn = np.argmax(S)
    if np.exp(S[assn]) > 0.8:
      W[i] = assn

  outdir_path = os.path.join(scratch_path, 'bins')
  mkdir_p(outdir_path)
  for k in xrange(K):
    out_path = os.path.join(outdir_path, '{}.bin.txt'.format(k)) 
    sel_rids = np.nonzero(W == k)
    with open(out_path, 'w') as fout:
      for rid in np.nditer(sel_rids):
        rid = int(rid)
        bcode = idx_rid_map[rid]
        fout.write('{}\n'.format(bcode))
  print 'unassigned', sum((W == -1))
    
#=========================================================================
# main
#=========================================================================
def main(argv):

  help_str = '''
  phase.py [cmd] <inbam path> <invcf path> <scratch path>
  '''

  if len(argv) < 5:
    print help_str
    sys.exit(1)
  cmd = argv[1]
  bam_path, vcf_path, scratch_path  = argv[2:]

  assert cmd in [
    'mkinputs',
    'phase',
    'mkoutputs',
  ]

  if cmd == 'mkinputs':
    make_inputs(bam_path, vcf_path, scratch_path)
  elif cmd == 'phase':
    phase(scratch_path)
  elif cmd == 'mkoutputs':
    make_outputs(scratch_path)
  else:
    assert False

if __name__ == '__main__':
  main(sys.argv)

