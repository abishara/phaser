from setuptools import setup, find_packages

setup(
  name = "phaser",
  version = "0.1",
  packages = find_packages(),
  entry_points = {
    'console_scripts' : ["phaser = phaser.phase:main"]
  },
  install_requires=[
    'scipy>=0.18',
    'numpy>=1.11.0',
    'pysam>=0.9',
    'h5py',
    'PyVCF',
    'pandas',
  ],
  author='alex bishara',
  description='barcode phaser',

)
