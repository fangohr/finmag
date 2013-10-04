import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Demag, Zeeman
from finmag.util.meshes import cylinder

#mesh = df.BoxMesh(0,0,0,30,30,3,10,10,1)
mesh = cylinder(200,3,3)


Ms = 8.6e5
sim = Simulation(mesh, Ms,unit_length=1e-9)
sim.set_m((1, 1, 1))

A = 1.3e-11
D = 4e-3
sim.add(Exchange(A))
sim.add(DMI(D))
sim.add(Zeeman((0, 0, 0.1*Ms)))

#sim.add(Demag())

def loop(final_time, steps=100):
    t = np.linspace(sim.t + 1e-12, final_time, steps)
    for i in t:
        sim.run_until(i)
        p = df.plot(sim.llg._m)
    df.interactive()

loop(5e-9)

