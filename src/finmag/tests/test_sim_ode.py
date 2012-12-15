import dolfin as df
import numpy as np
from finmag import Simulation as Sim
from finmag.energies import Zeeman

def test_sim_ode():

    mesh = df.Box(0, 0, 0, 2, 2, 2, 1, 1, 1)

    sim = Sim(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.1
    sim.set_m((1, 0, 0))

    H0=1e5
    zeeman=Zeeman((0,0,H0))
    sim.add(zeeman)

    times = np.linspace(0, 1e-11, 100)
    g=sim.gamma/(1+sim.alpha**2)*sim.alpha*H0

    for t in times:
        sim.run_until(t)
        m=sim.llg._m.vector().array()
        mz=m[-1]
        assert abs(mz-np.tanh(g*t))<1e-8

if __name__=="__main__":
     test_sim_ode()
