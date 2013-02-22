#!/usr/bin/env python

from IPython.nbformat.current import reads
from ipnbdoctest import test_notebook as reproduce_notebook
from glob import glob
import pytest

#ipynb_files = ['doc/ipython_notebooks_src/tutorial-using-ipython-notebook.ipynb',  # passes
#               #'doc/ipython_notebooks_src/tutorial-use-of-logging.ipynb',  # fails
#              ]

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

ipynb_files = sorted(glob(os.path.join(MODULE_DIR, '../doc/ipython_notebooks_src/*.ipynb'))

@pytest.mark.parametrize("ipynb", ipynb_files)
def test_reproduce_ipython_notebook(ipynb):
    print "testing %s" % ipynb
    with open(ipynb) as f:
        nb = reads(f.read(), 'json')
    reproduce_notebook(nb)
