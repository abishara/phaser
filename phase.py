import os
import sys
import random
from collections import defaultdict, Counter
import numpy as np

from mlib import util
#from mlib.fixedK_ve import phase, make_outputs
from mlib.fixedK_betas import phase, make_outputs

#=========================================================================
# main
#=========================================================================
def main(argv):

  help_str = '''
  phase.py [cmd] <inbam path> <invcf path> <scratch path> [threads]
  '''

  if len(argv) < 5:
    print help_str
    sys.exit(1)
  cmd = argv[1]
  bam_path, vcf_path, scratch_path  = argv[2:]
  if len(argv) == 6:
    num_threads = int(argv[5])
  else:
    num_threads = 1

  assert cmd in [
    'mkinputs',
    'phase',
    'mkoutputs',
  ]

  util.mkdir_p(scratch_path)
  if cmd == 'mkinputs':
    util.make_inputs(bam_path, vcf_path, scratch_path)
  elif cmd == 'phase':
    phase(scratch_path)
  elif cmd == 'mkoutputs':
    make_outputs(scratch_path)
  else:
    assert False

if __name__ == '__main__':
  main(sys.argv)

