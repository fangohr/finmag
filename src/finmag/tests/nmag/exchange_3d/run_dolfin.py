import os
import dolfin as df
from finmag import Simulation as Sim
from finmag.energies import Exchange, Zeeman

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_simulation():
    L = 3e-8; W = 1e-8; H = 1e-8
    mesh = df.BoxMesh(0, 0, 0, L, W, H, 10, 4, 4)

    Ms = 0.86e6 # A/m
    A = 1.3e-11 # J/m

    sim = Sim(mesh, Ms)
    sim.set_m(("2*x[0]/L - 1","2*x[1]/W - 1","1"), L=3e-8, H=1e-8, W=1e-8)
    sim.alpha = 0.1
    sim.add(Zeeman((Ms/2, 0, 0)))
    sim.add(Exchange(A))

    t = 0; dt = 1e-11; tmax = 1e-9 # s

    fh = open(os.path.join(MODULE_DIR, "averages.txt"), "w")
    while t <= tmax:
        mx, my, mz = sim.llg.m_average
        fh.write(str(t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")
        t += dt
        sim.run_until(t)
    fh.close()

if __name__ == "__main__":
    run_simulation()
