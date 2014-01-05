#!/usr/bin/env python

from IPython.nbformat.current import reads
from ipynbdoctest import test_notebook as reproduce_notebook
from glob import glob
import os
import sys
import pytest

#ipynb_files = ['doc/ipython_notebooks_src/tutorial-using-ipython-notebook.ipynb',  # passes
#               #'doc/ipython_notebooks_src/tutorial-use-of-logging.ipynb',  # fails
#              ]

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

print("sys.path: {}".format(sys.path))

ipynb_files = sorted(glob(os.path.join(MODULE_DIR, '../doc/ipython_notebooks_src/*.ipynb')))
print "============================================================"
print "Found {} .ipynb files:".format(len(ipynb_files))
print "\n".join(ipynb_files)
print "============================================================"

# This parameterization checks each .ipynb file in a
# separate test, which is nicer for debugging from
# within Jenkins.
@pytest.mark.parametrize("ipynb", ipynb_files)
def test_reproduce_ipython_notebook(ipynb):
    print "\n=====   =====   =====   =====   =====   =====   =====   ====="
    print "Testing notebook: '{}'".format(ipynb)
    with open(ipynb) as f:
        nb = reads(f.read(), 'json')
    reproduce_notebook(nb)
