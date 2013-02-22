#!/usr/bin/env python

from IPython.nbformat.current import reads
from ipnbdoctest import test_notebook as reproduce_notebook

ipynb_files = ['doc/ipython_notebooks_src/tutorial-using-ipython-notebook.ipynb',  # passes
               #'doc/ipython_notebooks_src/tutorial-use-of-logging.ipynb',  # fails
              ]

def test_reproduce_ipython_notebook():
    for ipynb in ipynb_files:
        print "testing %s" % ipynb
        with open(ipynb) as f:
            nb = reads(f.read(), 'json')
        reproduce_notebook(nb)
