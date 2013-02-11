import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI_Old, DMI
from finmag.util.meshes import cylinder

#mesh = df.BoxMesh(0,0,0,30e-9,30e-9,3e-9,10,10,1)
mesh =cylinder(20,3,3)
mesh.coordinates()[:]*=1e-9 #unit length doesn't work for DMI?

Ms = 8.6e5
sim = Simulation(mesh, Ms)
sim.set_m((1, 1, 1))

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

