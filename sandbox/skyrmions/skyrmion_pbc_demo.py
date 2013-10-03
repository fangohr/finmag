import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Zeeman

R1=460
R2=400
N=100
mesh = df.RectangleMesh(0,0,R1,R2,184,160)

def m_init_fun(pos):
    return np.random.random(3)-0.5

Ms = 8.6e5
sim = Simulation(mesh, Ms, pbc='2d',unit_length=1e-9)
sim.set_m(m_init_fun)

A = 1.3e-11
D = 4e-3
sim.add(Exchange(A))
sim.add(DMI(D))
sim.add(Zeeman((0,0,0.45*Ms)))

sim.alpha = 1
def loop(final_time, steps=500):
    t = np.linspace(sim.t + 1e-12, final_time, steps)
    for i in t:
        sim.run_until(i)
        p = df.plot(sim.llg._m)
        
    df.plot(sim.llg._m).write_png("vortex")
    df.interactive()
    
    
loop(100e-9)



