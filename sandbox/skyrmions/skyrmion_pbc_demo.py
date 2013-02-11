import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI_Old, DMI, Demag
from finmag.util.helpers import vector_valued_function

R=30
mesh = df.BoxMesh(0,0,0,R,R,3,10,10,1)

def m_init_fun(pos):
    if pos[0]<R/2.0:
        if pos[1]<R/2.0:
            return [-1,1,1]
        else:
            return [1,1,1]
    else:
        if pos[1]<R/2.0:
            return [1,-1,1]
        else:
            return [-1,-1,1]

m_init = vector_valued_function(m_init_fun, mesh)

Ms = 8.6e5
sim = Simulation(mesh, Ms, pbc2d=True,unit_length=1e-9)
sim.set_m((1,2,3))

A = 1.3e-11
D = 4e-3
sim.add(Exchange(A,pbc2d=True))
sim.add(DMI(D,pbc2d=False))
#sim.add(Demag())

def loop(final_time, steps=100):
    t = np.linspace(sim.t + 1e-12, final_time, steps)
    for i in t:
        sim.run_until(i)
        p = df.plot(sim.llg._m)
    df.interactive()

loop(1e-9)

