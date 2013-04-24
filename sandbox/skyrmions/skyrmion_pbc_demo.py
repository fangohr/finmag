import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Demag, Zeeman



R=150
N=50
#mesh = df.RectangleMesh(0,0,R,R,20,20)
mesh = df.RectangleMesh(0,0,R,R,N,N)

def m_init_fun(pos):
    return np.random.random(3)-0.5




Ms = 8.6e5
sim = Simulation(mesh, Ms, pbc='2d',unit_length=1e-9)
sim.set_m(m_init_fun)


#A = 3.57e-13
#D = 2.78e-3

A = 1.3e-11
D = 4e-3
sim.add(Exchange(A))
sim.add(DMI(D))
sim.add(Zeeman((0,0,0.35*Ms)))
#sim.add(Demag())

def loop(final_time, steps=100):
    t = np.linspace(sim.t + 1e-12, final_time, steps)
    for i in t:
        sim.run_until(i)
        p = df.plot(sim.llg._m)
        
    df.plot(sim.llg._m).write_png("vortex")
    df.interactive()
    
    
loop(1e-9)



