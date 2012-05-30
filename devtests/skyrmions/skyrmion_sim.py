import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI

mesh = df.Box(0,0,0,30e-9,30e-9,3e-9,10,10,1)
Ms = 8.6e5
sim = Simulation(mesh, Ms)
sim.set_m((Ms, 0, 0))

A = 1.3e-11
D = 4e-3
sim.add(Exchange(A))
sim.add(DMI(D))

series = df.TimeSeries("solution/m")
t = np.linspace(0, 1e-9, 1000)
for i in t:
    sim.run_until(i)
    p = df.plot(sim.llg._m)
    p.write_png("m_{}".format(i))
    series.store(sim.llg._m.vector(), i)
df.interactive()
