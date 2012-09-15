import matplotlib
matplotlib.use('Agg')

from numpy import array
from finmag import sim_with
from finmag.util.meshes import ellipsoid
import matplotlib.pyplot as plt
import sys
import logging
logger = logging.getLogger("finmag")

# This example essentially reproduces Example 2.3 in the nmag manual;
# see: http://nmag.soton.ac.uk/nmag/current/manual/html/example_hysteresis_ellipsoid/doc.html

r1 = 30.0
r2 = 10.0
r3 = 10.0
maxh = 3.0

Ms = 1e6         # A/m
A = 13.0e-12     # J/m
alpha = 1.0      # large damping for quick convergence
H = 1e6          # external field strength in A/m
m_init = (1, 0, 0)

H_max = 1e6  # maximum external field strength in A/m
direction = array([1.0, 0.01, 0.0])
N = 20

if len(sys.argv) > 1:
    N = int(sys.argv[1])
print "Using N={}".format(N)

if len(sys.argv) > 2:
    maxh = float(sys.argv[2])
print "Using maxh={}".format(maxh)

mesh = ellipsoid(r1, r2, r3, maxh)
sim = sim_with(mesh, Ms, m_init, alpha=alpha, unit_length=1e-9, A=A, demag_solver='FK')
print sim.mesh_info()

#(hvals, mvals) = sim.hysteresis_loop(H_max, direction, N, filename="snapshots/hysteresis_loop_example/hysteresis_ellipsoid.pvd", save_snapshots=True, save_every=10e-12, force_overwrite=True)
(hvals, mvals) = sim.hysteresis_loop(H_max, direction, N)
plt.plot(hvals, mvals,'o-', label='maxh={}'.format(maxh))
plt.ylim((-1.1, 1.1))
plt.title("Hysteresis loop: ellipsoid (r1={}, r2={}, r3={})".format(r1,r2,r3,maxh))
plt.legend(loc='best')
plt.savefig('plot_hysteresis_loop__maxh-{:04.1f}_N-{:03d}.pdf'.format(maxh, N))
