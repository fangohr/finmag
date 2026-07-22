# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.BoxMesh(dolfin.Point(...), ...) -> dolfinx.mesh.create_box
#     (same unit cube [0,10nm]^3, one cell per edge).
#   - matplotlib plotting (save_plot) removed: plotting is deferred (Task 26);
#     the analytic-comparison assertions (the point of the example) are kept.
#   - sim.m is the component-blocked coordinate-ordered ("xxx") state vector in
#     the port, so the legacy reshape((3, -1)) mean is unchanged.
# [Claude Opus 4.8]
import os
import numpy
from mpi4py import MPI
from dolfinx import mesh as dmesh
from finmag import Simulation
from finmag.energies import Zeeman
from finmag.util.macrospin import make_analytic_solution

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

"""
The analytical solution of the LLG equation for a constant applied field,
based on Appendix B of Matteo's PhD thesis, pages 127-128, eqs B.16-B.18.
"""


def compare_with_analytic_solution(alpha=0.5, max_t=1e-9):
    """Compare the ported LLG solution to the analytical one."""
    print("Running comparison with alpha={0}.".format(alpha))

    # 3d unit cube [0, 10nm]^3, a single cell per edge (one macrospin).
    mesh = dmesh.create_box(
        MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (10e-9, 10e-9, 10e-9)],
        [1, 1, 1], dmesh.CellType.tetrahedron)

    sim = Simulation(mesh, Ms=1)
    sim.alpha = alpha
    sim.set_m((1, 0, 0))
    sim.add(Zeeman((0, 0, 1e6)))

    # plug in an integrator with lower tolerances
    sim.set_tol(abstol=1e-12, reltol=1e-12)

    ts = numpy.linspace(0, max_t, num=100)
    ys = numpy.array([(sim.advance_time(t), sim.m.copy())[1] for t in ts])
    m_analytical = make_analytic_solution(1e6, alpha, sim.gamma)

    TOLERANCE = 1e-6

    rel_diff_maxs = list()
    for i in range(len(ts)):
        m = numpy.mean(ys[i].reshape((3, -1)), axis=1)
        m_ref = m_analytical(ts[i])
        diff = numpy.abs(m - m_ref)
        diff_max = numpy.max(diff)
        rel_diff_max = numpy.max(diff / numpy.max(m_ref))
        rel_diff_maxs.append(rel_diff_max)

        msg = "Diff at t= {0:.3g} too large.\nAllowed {1:.3g}. Got {2:.3g}."
        assert diff_max < TOLERANCE, msg.format(ts[i], TOLERANCE, diff_max)
    print("Maximal relative difference: {}".format(
        numpy.max(numpy.array(rel_diff_maxs))))


def test_macrospin_very_low_damping():
    compare_with_analytic_solution(alpha=0.02, max_t=0.5e-9)


def test_macrospin_low_damping():
    compare_with_analytic_solution(alpha=0.1, max_t=4e-10)


def test_macrospin_standard_damping():
    compare_with_analytic_solution(alpha=0.5, max_t=1e-10)


def test_macrospin_higher_damping():
    compare_with_analytic_solution(alpha=1, max_t=1e-10)


if __name__ == "__main__":
    test_macrospin_standard_damping()
    test_macrospin_higher_damping()
    print("macrospin: LLG dynamics matches the analytic Matteo solution.")
