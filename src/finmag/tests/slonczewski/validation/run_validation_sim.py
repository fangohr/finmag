import os
import dolfin as df
import numpy as np
from math import pi, sin, cos
from finmag import Simulation as Sim
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.util.consts import mu0
import logging
logger=logging.getLogger('finmag')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
averages_file = os.path.join(MODULE_DIR, "averages.txt")

def run_simulation(t_max = 10e-9):
    L = W = 12.5e-9; H = 5e-9;
    mesh = df.BoxMesh(0, 0, 0, L, W, H, 5, 5, 2)
    sim = Sim(mesh, Ms=8.6e5)
    sim.set_m((1, 0.01, 0.01))
    sim.alpha = 0.014
    sim.gamma = 221017

    H_app_mT = np.array([0.2, 0.2, 10.0])
    H_app_SI = H_app_mT / (1000 * mu0)
    sim.add(Zeeman(tuple(H_app_SI)))

    sim.add(Exchange(1.3e-11))
    sim.add(UniaxialAnisotropy(-1e5, (0, 0, 1)))

    I = 5e-5 # current in A
    J = I / (L * W) # current density in A/m^2
    theta = 40.0 * pi/180; phi = pi/2 # polarisation direction
    p = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    sim.llg.use_slonczewski(J=J, P=0.4, d=H, p=p)

    with open(averages_file, "w") as f:
        dt = 5e-12; 
        for t in np.arange(0, t_max, dt):
            sim.run_until(t)
            logger.info("integrated to time = %gs / %gs" % (t, t_max))
            f.write("{} {} {} {}\n".format(t, *sim.m_average))
    return len(np.arange(0, t_max, dt))


if __name__ == "__main__":
    run_simulation(short=True)
