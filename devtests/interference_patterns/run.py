import dolfin as df
import numpy as np
import math
from finmag import Simulation
from finmag.util.consts import mu0
from finmag.energies import Zeeman, Exchange, ThinFilmDemag

alpha = 0.012 # dimensionless
gamma = 2.210173e5 # m/(As), our value

# Permalloy.
Ms = 860e3 # A/m
A = 13.0e-12 # J/m, our value, paper uses D/(gamma*h_bar) without giving the number

# Mesh
# Use dolfin for debugging, use geo file later.
# The film described in figures 5 and 6, page 7 and section 5, page 8 is
# 440 nm * 440 nm * 5 nm. We'll put the center of the film at 0,0,0.

L = 440e-9
W = 440e-9
H = 5e-9
dL = L/2; dW = W/2; dH = H/2;

# Use exchange length to figure out discretisation.
l_ex = np.sqrt(2 * A / (mu0 * Ms**2))
print "The exchange length is l_ex={:.3}.".format(l_ex)

discretisation = math.floor(l_ex*1e9)*1e-9 
nx, ny, nz = (L/discretisation, W/discretisation, H/discretisation)
mesh = df.Box(-dL, -dW, -dH, dL, dW, dH, int(nx), int(ny), int(nz))

sim = Simulation(mesh, Ms)
sim.add(Zeeman((0, 0, 1.1 * Ms))) # section 3 and end of section 5
sim.add(Exchange(A))
sim.add(ThinFilmDemag())
sim.run_until(1e-10)
