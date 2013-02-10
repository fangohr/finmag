import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI_Old, DMI

mesh = df.BoxMesh(0,0,0,5e-9,5e-9,1e-9,2,2,1)

Ms = 8.6e5
sim = Simulation(mesh, Ms, pbc2d=True)
sim.set_m((1, 2, 10))

A = 1.3e-11
D = 4e-3
#sim.add(Exchange(A,pbc2d=True))
sim.add(DMI(D,pbc2d=True))

def loop(final_time, steps=100):
    t = np.linspace(sim.t + 1e-12, final_time, steps)
    for i in t:
        sim.run_until(i)
        p = df.plot(sim.llg._m)
    df.interactive()

loop(0.1e-9)

