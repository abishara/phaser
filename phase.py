import os
import sys
import random
from collections import defaultdict, Counter
import numpy as np
import click
import logging

from mlib import util
#from mlib.fixedK_ve import phase, make_outputs
from mlib.fixedK_betas import phase, make_outputs

#=========================================================================
# main
#=========================================================================
@click.group()
def main():
    pass

@main.command("make-inputs")
@click.option("--bam-path", help="The path to the input bam file", required=True, default=None)
@click.option("--vcf-path", help="Path to the call polymorphisms, in VCF format.", required=True, default=None)
@click.option("--work-path", help="Path for temporary files, created in the process.", required=True, default=None)
def make_inputs( bam_path, vcf_path, work_path):
  """
  Documentation for making inputs should be added here.
  """
  logging.info("Generating input.")
  util.mkdir_p(work_path)
  util.make_inputs(bam_path, vcf_path, work_path)
  logging.info("Input generation done.")

@main.command("phase")
@click.option("--work-path", help="Path for temporary files, created in the process.", required=True, default=None)
def phase(work_path):
  """
  Documentation for phasing should be added here.
  """
  logging.info("Phasing samples.")
  util.mkdir_p(work_path)
  phase(scratch_path)
  logging.info("Sample phasing done.")

@main.command("make-outputs")
@click.option("--work-path", help="Path for temporary files, created in the process.", required=True, default=None)
def make_outputs(work_path):
  """
  Documentation for making output should be added here.
  """
  logging.info("Generating output.")
  util.mkdir_p(work_path)
  make_outputs(scratch_path)
  logging.info("Output generation complete.")

if __name__ == '__main__':
  logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',level=logging.DEBUG,datefmt='%Y-%m-%d %H:%M:%S')  
  main()
