import dolfin as df
from finmag import Simulation


def _test_scipy_advance_time():
    mesh = df.UnitIntervalMesh(10)
    sim = Simulation(mesh, Ms=1, unit_length=1e-9, integrator_backend="scipy")
    sim.set_m((1, 0, 0))
    sim.advance_time(1e-12)
    sim.advance_time(2e-12)
    sim.advance_time(2e-12)
    sim.advance_time(2e-12)


def test_scipy_advance_time_zero_first():
    mesh = df.UnitIntervalMesh(10)
    sim = Simulation(mesh, Ms=1, unit_length=1e-9, integrator_backend="scipy")
    sim.set_m((1, 0, 0))
    sim.advance_time(0)
    sim.advance_time(1e-12)
    sim.advance_time(2e-12)
    sim.advance_time(2e-12)
    sim.advance_time(2e-12)

if __name__ == "__main__":
    test_scipy_advance_time_zero_first()
