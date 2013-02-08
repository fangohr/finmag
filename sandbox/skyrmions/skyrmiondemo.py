import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI_Old, DMI

mesh = df.Box(0,0,0,30e-9,30e-9,3e-9,10,10,1)

Ms = 8.6e5
sim = Simulation(mesh, Ms)
sim.set_m((Ms, 0, 0))

A = 1.3e-11
D = 4e-3
sim.add(Exchange(A))
sim.add(DMI(	D))

def loop(final_time, steps=100):
    t = np.linspace(sim.t + 1e-12, final_time, steps)
    for i in t:
        sim.run_until(i)
        p = df.plot(sim.llg._m)
    df.interactive()

loop(5e-10)

