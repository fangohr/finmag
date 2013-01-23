import os
import numpy as np
import pylab
import dolfin as df

from finmag import Simulation
from finmag.energies import UniaxialAnisotropy, Exchange, Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
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


def run_simulation():

    mu0   = 4.0 * np.pi * 10**-7  # vacuum permeability             N/A^2
    Ms    = 1.0e6                 # saturation magnetisation        A/m
    A     = 13.0e-12              # exchange coupling strength      J/m
    Km    = 0.5 * mu0 * Ms**2     # magnetostatic energy density scale   kg/ms^2
    lexch = (A/Km)**0.5           # exchange length                 m
    K1    = Km

    L = lexch # cube length in m
    nx = 10
    Lx = L*nx
    ny = 1
    Ly = ny*L
    nz = 30
    Lz = nz*L
    mesh = df.BoxMesh(0, 0, 0, Lx, Ly, Lz, nx, ny, nz)


    #anisotropy direction starts is [0,0,1] in lower half of the film
    #and [1,0,0] in upper half. This is a toy model of the exchange spring
    #systems that Bob Stamps is working on.
    expr_a = df.Expression(("x[2]<=Lz/2. ? 0 : 1","0", "x[2]<=Lz/2. ? 1 : 0"),Lz=Lz)

    #discretise material parameter as discontinous Galerkin order 0 basis function
    V3_DG0 = df.VectorFunctionSpace(mesh,"DG",0,dim=3)
    a = df.interpolate(expr_a,V3_DG0)

    sim = Simulation(mesh, Ms)
    sim.set_m((1,0,1))
    sim.add(UniaxialAnisotropy(K1, a))
    sim.add(Exchange(A))
    
    sim.relax()

    #create simple plot
    xpos=[]
    Mx = []
    Mz = []
    ax = []
    az = []

    aS3 = df.interpolate(a,sim.S3) 

    zs = np.linspace(0,Lz,200)
    for z in zs:
        pos = (Lx/2., Ly/2., z )
        Mx.append(sim.llg._m(pos)[0])
        Mz.append(sim.llg._m(pos)[2])
        ax.append(a(pos)[0])
        az.append(a(pos)[2])
        
    pylab.plot(Mx,zs/1e-9,'-',label='Magnetisation Mx')
    pylab.plot(Mz,zs/1e-9,'--',label='Magnetisation Mz')
    pylab.plot(ax,zs/1e-9,'-o',label='Anisotropy vector ax')
    pylab.plot(az,zs/1e-9,'-x',label='Anisotropy vector az')
    pylab.ylabel('position z in film [nm]')
    pylab.legend(loc='upper center')
    pylab.savefig(os.path.join(MODULE_DIR,'profile.png'))
    #pylab.show()

    ##this only works with an X-display, so comment out for jenkins
    #v = df.plot(sim.llg._m, 
    #            title='exchange spring across layers with different anisotropy directions',
    #            axes=True)
    #
    #v.elevate(-90) 
    #v.update(sim.llg._m)    # bring settings above into action
    ##v.write_png(os.path.join(MODULE_DIR,'exchangespring.png')) #this is broken in my dolfin, HF
    #                                                            #the bitmap looks random  
    #v.write_ps(os.path.join(MODULE_DIR,'exchangespring'))       #will write exchangespring.eps
    #os.system("ps2png exchangespring.eps exchangespring.png")   #create the png file for documentation

    f=df.File(os.path.join(MODULE_DIR,'exchangespring.pvd'))    #same more data for paraview
    f << sim.llg._m

    print("Written plots and data to %s" % (os.path.join(MODULE_DIR,'exchangespring.*')))
        
if __name__ == "__main__":
    run_simulation()






