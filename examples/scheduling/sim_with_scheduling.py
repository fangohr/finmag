# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.BoxMesh(dolfin.Point(...)) -> dolfinx.mesh.create_box.
# The scheduling API (sim.schedule with every/at/at_end) is unchanged (Task 12).
# [Claude Opus 4.8]
import logging
from mpi4py import MPI
from dolfinx import mesh as dmesh
from finmag import Simulation
from finmag.energies import Exchange, Demag, Zeeman

log = logging.getLogger(name="finmag")
log.setLevel(logging.ERROR)  # To better show output of this program.


def progress(s):
    print("We have integrated up to t = {:.3f} ns.".format(1e9 * s.t))
    print("Average magnetisation is m = {}.".format(s.m_average))


def halfway_done(s):
    print("We are halfway done!")


def done(s):
    print("Woohoo!")


def example_simulation():
    Ms = 8.6e5
    mesh = dmesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (40.0, 20.0, 20.0)],
        [10, 5, 5], dmesh.CellType.tetrahedron)

    example = Simulation(mesh, Ms, name="sim_with_scheduling")
    example.set_m((0.1, 1, 0))
    example.add(Exchange(13.0e-12))
    example.add(Demag())
    example.add(Zeeman((Ms / 2, 0, 0)))
    return example


if __name__ == "__main__":
    sim = example_simulation()

    t_final = 1.05e-9
    calls = {"progress": 0, "halfway": 0, "done": 0}
    sim.schedule(lambda s: calls.__setitem__("progress", calls["progress"] + 1) or progress(s),
                 every=1e-10, at_end=True)
    sim.schedule(lambda s: calls.__setitem__("halfway", calls["halfway"] + 1) or halfway_done(s),
                 at=t_final / 2)
    sim.schedule(lambda s: calls.__setitem__("done", calls["done"] + 1) or done(s),
                 at_end=True)

    sim.run_until(t_final)

    # Self-validation: every scheduled callback fired the expected number of times.
    assert calls["halfway"] == 1, "the `at` callback should fire exactly once"
    assert calls["done"] == 1, "the `at_end` callback should fire exactly once"
    assert calls["progress"] >= 10, "the `every` callback should fire repeatedly"
    print("scheduling: every/at/at_end callbacks fired as scheduled.")
