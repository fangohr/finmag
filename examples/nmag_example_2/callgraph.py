#!/usr/bin/env python

try:
    from pycallgraph import PyCallGraph
    from pycallgraph import Config
    from pycallgraph import GlobbingFilter
    from pycallgraph import Grouper
    from pycallgraph.output import GraphvizOutput
except ImportError:
    print "You need to install pycallgraph (for instance with `pip install pycallgraph`)."
    raise

from run_finmag import run_simulation

config = Config()
config.trace_grouper = Grouper(groups=[
    "finmag.util.*",
    "finmag.integrators.*",
    "finmag.energies.*",
    "finmag.sim.*",
])

# `max_depth`=15 is the level that would account for all calls to
# Exchange.__compute_field_petsc; 14 would miss those from TableWriter.save
config.trace_filter = GlobbingFilter(include=[
    'finmag.*',
    'run_finmag.*',
])
graphviz = GraphvizOutput(output_file='finmag_callgraph.png')

with PyCallGraph(output=graphviz, config=config):
    run_simulation()
