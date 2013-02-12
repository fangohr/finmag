import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI_Old, DMI, Demag,Zeeman
from finmag.util.helpers import vector_valued_function
from finmag.llb.sllg import SLLG


R=150
N=50
#mesh = df.RectangleMesh(0,0,R,R,20,20)
mesh = df.RectangleMesh(0,0,R,R,N,N)

def m_init_fun(pos):
    return np.random.random(3)-0.5


pbc=True

Ms = 8.6e5
sim = SLLG(mesh, Ms,unit_length=1e-9,pbc2d=pbc)
sim.set_m(m_init_fun)
sim.T=20
#A = 3.57e-13
#D = 2.78e-3

A = 1.3e-11
D = 4e-3
sim.add(Exchange(A,pbc2d=pbc))
sim.add(DMI(D,pbc2d=pbc))
sim.add(Zeeman((0,0,0.35*Ms)))
#sim.add(Demag())

def loop(final_time, steps=200):
    t = np.linspace(sim.t + 1e-12, final_time, steps)
    for i in t:
        sim.run_until(i)
        p = df.plot(sim._m)
    df.interactive()

loop(5e-10)

