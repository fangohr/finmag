import dolfin as df
from numpy import linspace
from math import cos, sin, pi
from finmag import sim_with
from finmag.util.meshes import ellipsoid

r1 = 30.0
r2 = 10.0
r3 = 10.0
maxh = 3.0

Ms = 1e6         # A/m
A = 13.0e-12     # J/m
alpha = 1.0      # large damping for quick convergence
H = 1e6          # external field strength in A/m
m_init = (1, 0, 0)

# Create a few external field values (at 45 degree angles
# to each other, sweeping a half-circle).
H_ext_list = [(cos(t)*H, sin(t)*H, 0.01*H) for t in linspace(0, pi, 5)]

mesh = ellipsoid(r1, r2, r3, maxh)
sim = sim_with(mesh, Ms, m_init, alpha=alpha, unit_length=1e-9, A=A, demag_solver='FK')

sim.hysteresis(H_ext_list[1:3], filename="snapshots/hysteresis_ellipsoid.pvd", save_snapshots=True, save_every=10e-12, force_overwrite=True)
