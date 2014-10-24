import os
import pylab
import numpy as np
import dolfin as df
from finmag.field import Field
from finmag import Simulation
from finmag.energies import UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
"""
Trying to test spatially varying anisotropy.
"""


def run_simulation(debug_plots=False):

    mu0 = 4.0 * np.pi * 10 ** -7  # vacuum permeability             N/A^2
    Ms = 1.0e6                 # saturation magnetisation        A/m
    A = 13.0e-12              # exchange coupling strength      J/m
    Km = 0.5 * mu0 * Ms ** 2     # magnetostatic energy density scale   kg/ms^2
    lexch = (A / Km) ** 0.5           # exchange length                 m
    K1 = Km

    L = lexch  # cube length in m
    nx = 20
    Lx = L * nx
    mesh = df.IntervalMesh(nx, 0, Lx)

    # anisotropy direction starts at [0,1,0] at x=0 and changes to [1,0,0] at
    # x=Lx, but keep normalised
    expr_a = df.Expression(("x[0]/sqrt(x[0]*x[0]+(Lx-x[0])*(Lx-x[0]))",
                            "(Lx-x[0])/sqrt(x[0]*x[0]+(Lx-x[0])*(Lx-x[0]))",
                            "0"), Lx=Lx)

    # descritise material parameter. Generally a discontinous Galerkin order 0 basis function
    # is a good choice here (see example in manual) but for the test we use
    # CG1 space.

    V3_DG0 = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
    a = Field(V3_DG0, expr_a)

    sim = Simulation(mesh, Ms)
    sim.set_m((1, 1, 0))
    sim.add(UniaxialAnisotropy(K1, a))
    sim.relax(stopping_dmdt=1)

    # create simple plot
    xpos = []
    Mx = []
    My = []
    Mz = []
    ax = []
    ay = []
    az = []

    xs = np.linspace(0, Lx, 200)
    for x in xs:
        pos = (x,)
        Mx.append(sim.m_field.probe(pos)[0])
        My.append(sim.m_field.probe(pos)[1])
        Mz.append(sim.m_field.probe(pos)[2])
        ax.append(a.probe(pos)[0])
        ay.append(a.probe(pos)[1])
        az.append(a.probe(pos)[2])

    if debug_plots:
        pylab.plot(xs, Mx, '-o', label='Mx')
        pylab.plot(xs, ax, '-x', label='ax')
        pylab.savefig(os.path.join(MODULE_DIR, 'profile.png'))
        print "Note that the alignment is pretty good everywhere, but not at x=0. Why?"
        print "It also seems that Mx is ever so slightly greater than ax -- why?"
        print "Uncomment the show() command to see this."
        # pylab.show()

    return sim, a, Mx, My, Mz, ax, ay, az


def test_spatially_varying_anisotropy_direction_a(tmpdir, debug=False):
    sim, a, Mx, My, Mz, ax, ay, az = run_simulation(debug)

    # Interpolate a on mesh of M
    diffx = (np.array(ax) - np.array(Mx))
    diffy = (np.array(ay) - np.array(My))
    diffz = (np.array(az) - np.array(Mz))
    maxdiffx = max(abs(diffx))
    maxdiffy = max(abs(diffy))
    maxdiffz = max(abs(diffz))
    
    assert maxdiffx < 0.04
    assert maxdiffy < 0.04
    assert maxdiffz < 0.04

    if debug:
        # Save field for debugging (will be stored in /tmp/pytest-USERNAME/)
        sim.m_field.save_pvd(os.path.join(os.chdir(str(tmpdir), 'test.pvd')))

if __name__ == "__main__":
    test_spatially_varying_anisotropy_direction_a()
