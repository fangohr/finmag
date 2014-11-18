import os
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from finmag.field import Field
from finmag import Simulation
from finmag.energies import UniaxialAnisotropy
from finmag.util.consts import bloch_parameter

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_spatially_varying_anisotropy_axis(tmpdir, debug=False):
    Ms = 1e6
    A = 1.3e-11
    K1 = 6e5
    lb = bloch_parameter(A, K1)

    unit_length = 1e-9
    nx = 20
    Lx = nx * lb / unit_length
    mesh = df.IntervalMesh(nx, 0, Lx)

    # anisotropy axis goes from (0, 1, 0) at x=0 to (1, 0, 0) at x=Lx
    expr_a = df.Expression(("x[0] / sqrt(pow(x[0], 2) + pow(Lx-x[0], 2))",
                            "(Lx-x[0]) / sqrt(pow(x[0], 2) + pow(Lx-x[0], 2))",
                            "0"), Lx=Lx)
    # in theory, a discontinuous Galerkin (constant over the cell) is a good
    # choice to represent material parameters. In this case though, the
    # parameter varies linearly, so we use the usual CG.
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    a = Field(V, expr_a)

    sim = Simulation(mesh, Ms, unit_length)
    sim.set_m((1, 1, 0))
    sim.add(UniaxialAnisotropy(K1, a))
    sim.relax()

    # probe the easy axis and the magnetisation along the interval
    points = 100
    xs = np.linspace(0, Lx, points)
    axis_xs = np.zeros((points, 3))
    m_xs = np.zeros((points, 3))

    for i, x in enumerate(xs):
        axis_xs[i] = a(x)
        m_xs[i] = sim.m_field(x)

    # we want to the magnetisation to follow the easy axis
    # it does so, except at x=0, what is happening there?
    diff = np.abs(m_xs - axis_xs)
    assert diff.max() < 0.02

    if debug:
        old = os.getcwd()
        os.chdir(tmpdir)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xs, axis_xs[:, 0], "b+", label="a_x")
        ax.plot(xs, m_xs[:, 0], "r--", label="m_x")
        ax.legend(loc="upper left")
        ax.set_ylim((0, 1.05))
        ax.set_xlabel("x (nm)")
        plt.savefig('spatially_varying_easy_axis.png')
        plt.close()
        sim.m_field.save_pvd('spatially_varying_easy_axis.pvd')
        os.chdir(old)

if __name__ == "__main__":
    test_spatially_varying_anisotropy_axis(".", debug=True)
