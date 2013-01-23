import os
import time
import logging
import numpy as np
import dolfin as df
from scipy.optimize import bisect
from finmag import Simulation
from finmag.energies import UniaxialAnisotropy, Exchange, Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

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
energies = dict()

def run_simulation(lfactor, m_init, m_init_name=""):
    L = lfactor * lexch
    divisions = int(round(lfactor * 2)) # that magic number influences L
    mesh = df.BoxMesh(0, 0, 0, L, L, L, divisions, divisions, divisions)

    exchange = Exchange(A)
    anisotropy = UniaxialAnisotropy(K1, [0, 0, 1])
    demag = Demag()

    sim = Simulation(mesh, Ms)
    sim.set_m(m_init)
    sim.add(exchange)
    sim.add(anisotropy)
    sim.add(demag)
    sim.relax()

    # Save average magnetisation.
    mx, my, mz = sim.llg.m_average
    with open(os.path.join(MODULE_DIR, "data_m.txt"), "a") as f:
        t = time.asctime()
        f.write("{} {} {} {} {} {}\n".format(m_init_name, lfactor, mx, my, mz, t))

    # Save energies.
    # We could call sim.total_energy, but we want the individual contributions.
    e_exc   = exchange.compute_energy() / (sim.Volume * Km)
    e_anis  = anisotropy.compute_energy() / (sim.Volume * Km)
    e_demag = demag.compute_energy() / (sim.Volume * Km)
    e_total = e_exc + e_anis + e_demag # relative total energy density

    with open(os.path.join(MODULE_DIR, "data_energies.txt"), "a") as f:
        t = time.asctime()
        f.write("{} {} {} {} {} {} {}\n".format(m_init_name,
            lfactor, e_total, e_exc, e_anis, e_demag, t))

    return e_total

def energy_difference(lfactor):
    print "Running the two simulations for lfactor={}.".format(lfactor)
    e_vortex = run_simulation(lfactor, vortex_init, "vortex")
    e_flower = run_simulation(lfactor, flower_init, "flower")
    diff = e_vortex - e_flower
    with open(os.path.join(MODULE_DIR, "data_diffs.txt"), "a") as f:
        t = time.asctime()
        f.write("{} {} {} {} {}\n".format(lfactor, e_vortex, e_flower, diff, t))
    return diff

if __name__ == "__main__":
    print "Running standard problem 3."
    single_domain_limit = bisect(energy_difference, 8, 8.5, xtol=0.1)
    print "L = " + str(single_domain_limit) + "."

    from table_for_doc import write_table
    write_table()
