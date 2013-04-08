import os
import numpy as np
from math import pi, sin, cos
from finmag import Simulation as Sim
from finmag.energies import Exchange, Zeeman
from finmag.util.meshes import from_geofile
from finmag.util.consts import mu0

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
averages_file = os.path.join(MODULE_DIR, "averages.txt")
mesh = from_geofile(os.path.join(MODULE_DIR, "mesh.geo"))

def run_simulation():
    L = W = 12.5e-9; H = 2.5e-9;
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9)
    sim.set_m((1, 0.01, 0.01))
    sim.alpha = 0.014
    sim.gamma = 221017

    H_app_mT = np.array([0.0, 0.0, 10.0])
    H_app_SI = H_app_mT / (1000 * mu0)
    sim.add(Zeeman(tuple(H_app_SI)))

    sim.add(Exchange(1.3e-11))

    I = 5e-5 # current in A
    J = I / (L * W) # current density in A/m^2
    theta = 40.0 * pi/180; phi = pi/2 # polarisation direction
    p = (sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta))
    sim.llg.use_slonczewski(J=J, P=0.4, d=H, p=p)

    with open(averages_file, "w") as f:
        dt = 10e-12; t_max = 10e-9;
        for t in np.arange(0, t_max, dt):
            sim.run_until(t)
            f.write("{} {} {} {}\n".format(t, *sim.m_average))

if __name__ == "__main__":
    run_simulation()
