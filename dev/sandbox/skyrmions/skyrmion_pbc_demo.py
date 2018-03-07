import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Zeeman


def m_init_fun(pos):
    return np.random.random(3)-0.5


def relax(mesh):

    Ms = 8.6e5
    sim = Simulation(mesh, Ms, pbc='2d',unit_length=1e-9)
    sim.set_m(m_init_fun)

    sim.add(Exchange(1.3e-11))
    sim.add(DMI(D = 4e-3))
    sim.add(Zeeman((0,0,0.45*Ms)))

    sim.alpha = 0.5

    ts = np.linspace(0, 1e-9, 101)
    for t in ts:
        sim.run_until(t)
        p = df.plot(sim.llg._m)
    sim.save_vtk()
        
    df.plot(sim.llg._m).write_png("vortex")
    df.interactive()
    
    
    
if __name__=='__main__':
    mesh = df.RectangleMesh(0,0,100,100,40,40)
    relax(mesh)
    



