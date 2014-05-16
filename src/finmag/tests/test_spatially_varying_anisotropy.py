import os
import pylab
import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
"""
Trying to test spatially varying anisotropy.
"""


def run_simulation():

    mu0   = 4.0 * np.pi * 10**-7  # vacuum permeability             N/A^2
    Ms    = 1.0e6                 # saturation magnetisation        A/m
    A     = 13.0e-12              # exchange coupling strength      J/m
    Km    = 0.5 * mu0 * Ms**2     # magnetostatic energy density scale   kg/ms^2
    lexch = (A/Km)**0.5           # exchange length                 m
    K1    = Km

    L = lexch # cube length in m
    nx = 20
    Lx = L*nx
    mesh = df.IntervalMesh(nx, 0, Lx)

    #anisotropy direction starts at [0,1,0] at x=0 and changes to [1,0,0] at x=Lx, but keep normalised
    expr_a = df.Expression(("x[0]/sqrt(x[0]*x[0]+(Lx-x[0])*(Lx-x[0]))",
                            "(Lx-x[0])/sqrt(x[0]*x[0]+(Lx-x[0])*(Lx-x[0]))",
                            "0"),Lx=Lx)


    #descritise material parameter. Generally a discontinous Galerkin order 0 basis function
    #is a good choice here (see example in manual) but for the test we use CG1 space.

    V3_CG1 = df.VectorFunctionSpace(mesh,"CG",1,dim=3)
    a = df.interpolate(expr_a,V3_CG1)

    sim = Simulation(mesh, Ms)
    sim.set_m((1,1,0))
    sim.add(UniaxialAnisotropy(K1, a))
    sim.relax(stopping_dmdt=1)

    #create simple plot
    xpos=[]
    Mx = []
    ax = []

    xs = np.linspace(0,Lx,200)
    for x in xs:
        pos = (x,)
        Mx.append(sim._m(pos)[0])
        ax.append(a(pos)[0])

    pylab.plot(xs,Mx,'-o',label='Mx')
    pylab.plot(xs,ax,'-x',label='ax')
    pylab.savefig(os.path.join(MODULE_DIR,'profile.png'))
    print "Note that the alignment is pretty good everywhere, but not at x=0. Why?"
    print "It also seems that Mx is ever so slightly greater than ax -- why?"
    print "Uncomment the show() command to see this."
    #pylab.show()

    return sim,a


def test_spatially_varying_anisotropy_direction_a():

    sim,a = run_simulation()

    #interpolate a on mesh of M
    diff = (a.vector().array() - sim.m)
    maxdiff = max(abs(diff))
    print "maxdiff=",maxdiff
    print "The fairly large error seems to come from x=0. Why?"
    assert maxdiff < 0.018

    if __name__ == "__main__":

        f=df.File('test.pvd')
        f << sim._m

if __name__ == "__main__":
    test_spatially_varying_anisotropy_direction_a()






