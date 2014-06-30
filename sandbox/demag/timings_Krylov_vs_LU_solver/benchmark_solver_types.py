#!/usr/bin/env python

import dolfin as df
import numpy as np
import sys

try:
    solver_type = sys.argv[1]
except IndexError:
    print("Usage: time_solver_types.py SOLVER_TYPE [N]")
    sys.exit()

try:
    N = int(sys.argv[2])
except IndexError:
    N = 3

from finmag.example import barmini, bar
from finmag.example.normal_modes import disk

timings = []
for i in xrange(N):
    print("[DDD] Starting run #{}".format(i))
    #sim = bar(demag_solver_type=solver_type)
    sim = disk(d=100, h=10, maxh=3.0, relaxed=False, demag_solver_type=solver_type)
    df.tic()
    sim.relax()
    timings.append(df.toc())
    print("Latest run took {:.2f} seconds.".format(timings[-1]))
print("Timings (in seconds): {}".format(['{:.2f}'.format(t) for t in timings]))
print("Mean: {:.2f}".format(np.mean(timings)))
print("Median: {:.2f}".format(np.median(timings)))
print("Fastest: {:.2f}".format(np.min(timings)))
