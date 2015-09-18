import os
import dolfin as df
import numpy as np
from math import pi, sin, cos
from finmag import Simulation as Sim
from finmag.energies import Exchange, UniaxialAnisotropy, Zeeman
from finmag.util.consts import mu0

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
averages_file = os.path.join(MODULE_DIR, "averages.txt")


def run_simulation():
    L = W = 12.5e-9
    H = 5e-9
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(L, W, H), 5, 5, 2)
    sim = Sim(mesh, Ms=860e3, name="finmag_validation")
    sim.set_m((1, 0.01, 0.01))
    sim.alpha = 0.014
    sim.gamma = 221017

    H_app_mT = np.array([0.2, 0.2, 10.0])
    H_app_SI = H_app_mT / (1000 * mu0)
    sim.add(Zeeman(tuple(H_app_SI)))
    sim.add(Exchange(1.3e-11))
    sim.add(UniaxialAnisotropy(-1e5, (0, 0, 1)))

    I = 5e-5  # current in A
    J = I / (L * W)  # current density in A/m^2
    theta = 40.0 * pi / 180  # polarisation direction
    phi = pi / 2
    p = (sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta))
    sim.set_stt(current_density=J, polarisation=0.4, thickness=H, direction=p)

    sim.schedule("save_averages", every=5e-12)
    sim.run_until(10e-9)


if __name__ == "__main__":
    run_simulation()
