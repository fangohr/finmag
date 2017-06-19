"""
Demonstrating spatially varying anisotropy. Example with anisotropy vectors as follows:

-----------------------------------

--> --> --> --> --> --> --> --> -->
--> --> --> --> --> --> --> --> -->
--> --> --> --> --> --> --> --> -->

-----------------------------------

^  ^  ^  ^  ^  ^  ^  ^  ^  ^  ^  ^
|  |  |  |  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |  |  |  |

-----------------------------------
"""
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

import pylab
import dolfin as df

import matplotlib.pyplot as plt

from finmag import Simulation
from finmag.field import Field
from finmag.energies import UniaxialAnisotropy, Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_simulation(plot=False):
    mu0 = 4.0 * np.pi * 10**-7  # vacuum permeability N/A^2
    Ms = 1.0e6  # saturation magnetisation A/m
    A = 13.0e-12  # exchange coupling strength J/m
    Km = 0.5 * mu0 * Ms**2  # magnetostatic energy density scale kg/ms^2
    lexch = (A/Km)**0.5  # exchange length m
    unit_length = 1e-9
    K1 = Km

    L = lexch / unit_length
    nx = 10
    Lx = nx * L
    ny = 1
    Ly = ny * L
    nz = 30
    Lz = nz * L
    mesh = df.BoxMesh(df.Point(0, 0, 0), df.Point(Lx, Ly, Lz), nx, ny, nz)

    # Anisotropy easy axis is (0, 0, 1) in the lower half of the film and
    # (1, 0, 0) in the upper half. This is a toy model of the exchange spring
    # systems that Bob Stamps is working on.
    boundary = Lz / 2.0
    expr_a = df.Expression(("x[2] <= b ? 0 : 1", "0", "x[2] <= b ? 1 : 0"), b=boundary, degree=1)
    V = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
    a = Field(V, expr_a)

    sim = Simulation(mesh, Ms, unit_length)
    sim.set_m((1, 0, 1))
    sim.add(UniaxialAnisotropy(K1, a))
    sim.add(Exchange(A))
    sim.relax()

    if plot:
        points = 200
        zs = np.linspace(0, Lz, points)
        axis_zs = np.zeros((points, 3))  # easy axis probed along z-axis
        m_zs = np.zeros((points, 3))  # magnetisation probed along z-axis

        for i, z in enumerate(zs):
            axis_zs[i] = a((Lx/2.0, Ly/2.0, z))
            m_zs[i] = sim.m_field((Lx/2.0, Ly/2.0, z))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(zs, axis_zs[:, 0], "-o", label="a_x")
        ax.plot(zs, axis_zs[:, 2], "-x", label="a_z")
        ax.plot(zs, m_zs[:, 0], "-", label="m_x")
        ax.plot(zs, m_zs[:, 2], "-", label="m_z")
        ax.set_xlabel("z (nm)")
        ax.legend(loc="upper left")
        plt.savefig(os.path.join(MODULE_DIR, "profile.png"))
        sim.m_field.save_pvd(os.path.join(MODULE_DIR, 'exchangespring.pvd'))

if __name__ == "__main__":
    run_simulation(plot=True)
