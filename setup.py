# This file is part of gLund by S. Carrazza and F. A. Dreyer

from __future__ import print_function
import sys
from setuptools import setup, find_packages

if sys.version_info < (3,6):
    print("gLund requires Python 3.6 or later", file=sys.stderr)
    sys.exit(1)

with open('README.md') as f:
    long_desc = f.read()

setup(name= "glund",
      version = '0.0.1',
      description = "Generative models for jet substructure ",
      author = "F. Dreyer, S. Carrazza",
      author_email = "frederic.dreyer@cern.ch, stefano.carrazza@cern.ch",
      url="https://gitlab.cern.ch/JetsGame/gLund",
      long_description = long_desc,
      entry_points = {'console_scripts':
                      ['glund = glund.scripts.glund:main',
                       'glund_plot_samples = glund.scripts.glund_plot_samples:main',
                       'glund_test_encoding = glund.scripts.glund_test_encoding:main']},
      package_dir = {'': 'src'},
      packages = find_packages('src'),
      zip_safe = False,
      classifiers=[
            'Operating System :: Unix',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Physics',
            ],
     )
