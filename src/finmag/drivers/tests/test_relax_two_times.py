import dolfin as df
from finmag import Simulation
from finmag.energies import Zeeman

def test_relax_two_times():
    """
    Test whether we can call the relax method on Sim two times in a row.

    """
    mesh = df.BoxMesh(0, 0, 0, 10, 10, 10, 2, 2, 2)
    Ms = 0.86e6

    sim = Simulation(mesh, Ms)
    sim.set_m((1, 0, 0))

    external_field = Zeeman((0, Ms, 0))
    sim.add(external_field)
    sim.relax()
    t0 = sim.t # time needed for first relaxation

    external_field.set_value((0, 0, Ms))
    sim.relax()
    t1 = sim.t - t0 # time needed for second relaxation

    assert sim.t > t0
    assert abs(t1 - t0) < 1e-10
