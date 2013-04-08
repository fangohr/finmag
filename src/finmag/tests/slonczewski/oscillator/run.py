import os
import numpy as np
from finmag import Simulation as Sim
from finmag.energies import Exchange, Demag
from finmag.util.meshes import from_geofile

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
initial_m_file = os.path.join(MODULE_DIR, "m0.txt")
averages_file = os.path.join(MODULE_DIR, "averages.txt")
mesh = from_geofile(os.path.join(MODULE_DIR, "mesh.geo"))
mesh_centre = (5, 50, 50)
Ms = 8.6e5

def m_gen(rs):
    v = np.zeros(rs.shape)
    v[0] = 1
    v[1] = rs[2] - mesh_centre[2]
    v[2] = - (rs[1] - mesh_centre[1])
    return v

def create_initial_state():
    print "Creating initial relaxed state."
    sim = Sim(mesh, Ms=Ms, unit_length=1e-9)
    sim.set_m(m_gen)
    sim.alpha = 0.5
    sim.add(Exchange(1.3e-11))
    sim.add(Demag())
    sim.relax()
    np.savetxt(initial_m_file, sim.m)

def run_simulation():
    print "Running simulation of STT dynamics."
    sim = Sim(mesh, Ms=Ms, unit_length=1e-9)
    sim.set_m(np.loadtxt(initial_m_file))
    sim.alpha = 0.01
    sim.add(Exchange(1.3e-11))
    sim.add(Demag())
    sim.llg.use_slonczewski(J=0.1e12, P=0.4, d=10e-9, p=(0, 1, 0))
    with open(averages_file, "w") as f:
        dt = 5e-12; t_max = 10e-9;
        for t in np.arange(0, t_max, dt):
            sim.run_until(t)
            f.write("{} {} {} {}\n".format(t, *sim.m_average))

if __name__ == "__main__":
    if not os.path.exists(initial_m_file):
        create_initial_state()
    run_simulation()
