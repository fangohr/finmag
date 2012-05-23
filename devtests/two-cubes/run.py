import os
import numpy as np
import dolfin as df
from finmag.util.convert_mesh import convert_mesh
from finmag import Simulation
from finmag.energies import UniaxialAnisotropy, Exchange, Demag


MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

mesh = df.Mesh(convert_mesh(MODULE_DIR + "two-cubes.geo"))



#df.plot(mesh)
#
#df.interactive()


Ms    = 0.86e6                # saturation magnetisation        A/m
A     = 13.0e-12              # exchange coupling strength      J/m

init = (0, 0.1, 1)

sim = Simulation(mesh, Ms, unit_length=1e-9)
sim.add(Demag())
sim.add(Exchange(A))
sim.set_m(init)

f=df.File(os.path.join(MODULE_DIR,'cubes.pvd'))    #same more data for paraview

ns=1e-9
dt=0.01*ns
v=df.plot(sim.llg._m)
for time in np.arange(0,150.5*dt,dt):
    print "time=",time,"m=",
    print sim.llg.m_average
    sim.run_until(time)
    v.update(sim.llg._m)

    f << sim.llg._m



df.interactive()





