import os
import numpy as np
from scipy.optimize import bisect, curve_fit

"""
The poor man's version of nnag's solution. We know the single domain limit
is somewhere beetween 8 and 9 times lexch, so we compute the energies for 
those two values, fit a straight line and compute the bisection of those
two lines.

Of course, fitting a straight line to that is really brutal, so we are just
checking if we are even in the same ballpark than the correct solution.

"""

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

energies_per_state = np.genfromtxt(MODULE_DIR + "rel_e_densities.txt")

def f(x, a, b):
    return a * x + b

xs = np.array([8, 9])
ys_flower = energies_per_state[0]
ys_vortex = energies_per_state[1]

p_flower, _ = curve_fit(f, xs, ys_flower)
p_vortex, _ = curve_fit(f, xs, ys_vortex)

def diff(x):
    return (p_flower[0] - p_vortex[0]) * x + (p_flower[1] - p_vortex[1])

single_domain_limit = bisect(diff, 8, 9, xtol=0.1)
single_domain_limit_nmag = 8.461

print "limit L={}*lexch with diff={}.".format(
        single_domain_limit, diff(single_domain_limit))
print "vs. nmag L={}*lexch".format(single_domain_limit_nmag)
print "which would have a diff={}.".format(diff(single_domain_limit_nmag))
print "Please read the warning in the source code."

