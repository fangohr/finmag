import dolfin as df
from finmag import Simulation as Sim
from finmag.energies import Exchange, Demag

epsilon = 1e-16

def test_current_time():
    size = 20e-9
    simplices = 4
    mesh = df.BoxMesh(0, 0, 0, size, size, size, simplices, simplices, simplices)

    Ms = 860e3
    A = 13.0e-12

    sim = Sim(mesh, Ms)
    sim.set_m((1, 0, 0))
    sim.add(Exchange(A))
    sim.add(Demag())

    t = 0.0; t_max = 1e-10; dt = 1e-12;

    while t <= t_max:
        t += dt
        sim.run_until(t)
        # cur_t is equal to whatever time the integrator decided to probe last
        assert not sim.integrator.cur_t == 0.0
        # t is equal to our current simulation time
        assert abs(sim.t - t) < epsilon
