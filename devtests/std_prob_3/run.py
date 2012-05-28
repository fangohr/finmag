import os
import numpy as np
import dolfin as df

from finmag import Simulation
from finmag.energies import UniaxialAnisotropy, Exchange, Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"
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

flower_init = (0, 0, 1)
def vortex_init(rs):
    """
    from nmag's solution
        http://magnonics.ex.ac.uk:3000/attachments/723/run.py
    which cites Guslienko et al. APL 78 (24)
        http://link.aip.org/link/doi/10.1063/1.1377850

    """
    xs, ys, zs = rs
    rho = xs**2 + ys**2
    phi = np.arctan2(zs, xs)
    b = 2 * lexch
    m_phi = np.sin(2 * np.arctan(rho/b))
    return np.array([np.sqrt(1.0 - m_phi**2), m_phi*np.cos(phi), -m_phi*np.sin(phi)])

energies_per_state = []
for m_init in [flower_init, vortex_init]:
    energies_per_lfactor = [] 
    for lfactor in [8, 9]:
        L = lfactor * lexch # cube length in m
        divisions = lfactor * 2
        mesh = df.Box(0, 0, 0, L, L, L, divisions, divisions, divisions)

        sim = Simulation(mesh, Ms)
        sim.add(UniaxialAnisotropy(K1, [0, 0, 1]))
        sim.set_m(m_init)
        sim.add(Demag())
        sim.add(Exchange(A))

        sim.relax()

        total_energy_density = sim.total_energy()/sim.Volume
        relative_total_energy_density = total_energy_density/Km
        energies_per_lfactor.append(relative_total_energy_density)
        print "relative E_tot density {} for L={}*lexch.".format(
                relative_total_energy_density, lfactor)
    energies_per_state.append(energies_per_lfactor)

np.savetxt(MODULE_DIR + "rel_e_densities.txt", energies_per_state)
