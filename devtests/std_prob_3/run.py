import numpy as np
import dolfin as df

from finmag import Simulation
from finmag.energies.anisotropy import UniaxialAnisotropy
from finmag.energies.exchange import Exchange

"""
Micromag Standard Problem #3

specification:
    http://www.ctcms.nist.gov/~rdm/mumag.org.html
solution with nmag:
    http://magnonics.ex.ac.uk:3000/wiki/dynamag/Mumag3_nmag
a good write-up (Rave 1998):
    http://www.sciencedirect.com/science/article/pii/S030488539800328X

"""

mu0   = 4.0 * np.pi * 10**-7  # vacuum permeability             N/A^2
Ms    = 1.0e6                 # saturation magnetisation        A/m
A     = 13.0e-12              # exchange coupling strength      J/m
Km    = 0.5 * mu0 * Ms**2     # magnetostatic energy density    kg/ms^2
lexch = (A/Km)**0.5           # exchange length                 m
K1    = 0.1 * Km

for lfactor in range(1, 10):
    L = lfactor * lexch       # cube length                     m
    divisions = lfactor * 2
    mesh = df.Box(0, 0, 0, L, L, L, divisions, divisions, divisions)

    sim = Simulation(mesh, Ms)
    sim.add(UniaxialAnisotropy(K1, [0, 0, 1]))
    sim.add(Exchange(A))
