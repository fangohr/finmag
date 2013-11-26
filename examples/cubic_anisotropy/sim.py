import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, CubicAnisotropy
from finmag.util.consts import mu0

mesh = df.UnitCubeMesh(0, 0, 0, 1, 1, 40, 1, 1, 40)

Ms = 876626  # A/m
A = 1.46e-11  # J/m

K1 = -8608726
K2 = -13744132
K3 =  1100269
u1 = (0, -0.7071, 0.7071)
u2 = (0,  0.7071, 0.7071)
u3 = (-1, 0, 0)  # perpendicular to u1 and u2

# specification of fields close to oommf reference cubicEight_100pc.mif
# on http://www.southampton.ac.uk/~fangohr/software/oxs_cubic8.html

fields = np.zeros((250, 3))
fields[:, 0] = 5
fields[:, 1] = 5
fields[:, 2] = np.linspace(20000, -20000, 250)
fields = fields * 0.001 / mu0  # mT to A/m

sim = Simulation(mesh, Ms, unit_length=1e-9)
sim.set_m((0, 0, 1))
sim.add(Exchange(A))
sim.add(CubicAnisotropy(K1, u1, K2, u2, K3, u3))

mxs = sim.hysteresis(fields, lambda sim: sim.m_average[0])
np.savetxt("mxs", mxs)

