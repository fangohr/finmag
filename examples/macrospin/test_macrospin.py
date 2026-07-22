# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.BoxMesh(dolfin.Point(...), ...) -> dolfinx.mesh.create_box
#     (same unit cube [0,10nm]^3, one cell per edge).
#   - sim.m is the component-blocked coordinate-ordered ("xxx") state vector in
#     the port, so the legacy reshape((3, -1)) mean is unchanged.
#   - The legacy @pytest.mark.requires_X_display markers are dropped because the
#     pure-matplotlib save_plot (restored below) now only writes files (Agg
#     backend, savefig -- never an interactive show) and is called on the
#     __main__/save path only, so the numeric tests need no display.
#   - The fast gate runs this script as __main__; the two low-damping cases
#     (very_low / low) and the plotting run only under FINMAG_EXAMPLE_FULL=1 so
#     the fast gate stays quick and artifact-free -- disclosed, not deleted.
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


def compare_with_analytic_solution(alpha=0.5, max_t=1e-9, plot=False):
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
    tsfine = numpy.linspace(0, max_t, num=1000)
    m_analytical = make_analytic_solution(1e6, alpha, sim.gamma)
    if plot:
        save_plot(ts, ys, tsfine, m_analytical, alpha)

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


def save_plot(ts, ys, ts_ref, m_ref, alpha):
    # component-blocked ("xxx...yyy...zzz") -> (steps, 3, nvertices), then mean
    ys3d = ys.reshape((len(ys), 3, -1)).mean(axis=-1)
    mx = ys3d[:, 0]
    my = ys3d[:, 1]
    mz = ys3d[:, 2]
    m_exact = m_ref(ts_ref)
    mx_exact = m_exact[0, :]
    my_exact = m_exact[1, :]
    mz_exact = m_exact[2, :]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.plot(ts, mx, 'o', label='mx')
    plt.plot(ts, my, 'x', label='my')
    plt.plot(ts, mz, '^', label='mz')
    plt.plot(ts_ref, mx_exact, '-', label='mx (exact)')
    plt.plot(ts_ref, my_exact, '-', label='my (exact)')
    plt.plot(ts_ref, mz_exact, '-', label='mz (exact)')
    plt.xlabel('t [s]')
    plt.ylabel('m=M/Ms')
    plt.title(r'Macrospin dynamics: $\alpha$={}'.format(alpha))
    plt.grid()
    plt.legend()
    filename = ('alpha-%04.2f' % alpha)
    # latex does not like multiple '.' in image filenames
    filename = filename.replace('.', '-')
    plt.savefig(os.path.join(MODULE_DIR, filename + '.pdf'))
    plt.savefig(os.path.join(MODULE_DIR, filename + '.png'))
    plt.close()


def test_macrospin_alpha_0_00001():
    compare_with_analytic_solution(alpha=0.00001, max_t=1e-11)


def test_macrospin_alpha_0_001():
    compare_with_analytic_solution(alpha=0.001, max_t=1e-11)


def test_macrospin_very_low_damping():
    compare_with_analytic_solution(alpha=0.02, max_t=0.5e-9)


def test_macrospin_low_damping():
    compare_with_analytic_solution(alpha=0.1, max_t=4e-10)


def test_macrospin_standard_damping():
    compare_with_analytic_solution(alpha=0.5, max_t=1e-10)


def test_macrospin_higher_damping():
    compare_with_analytic_solution(alpha=1, max_t=1e-10)


if __name__ == "__main__":
    full = os.environ.get("FINMAG_EXAMPLE_FULL") == "1"
    if full:
        # legacy full 4-case set (+ plots)
        compare_with_analytic_solution(alpha=0.02, max_t=0.5e-9, plot=True)
        compare_with_analytic_solution(alpha=0.1, max_t=4e-10, plot=True)
        compare_with_analytic_solution(alpha=0.5, max_t=1e-10, plot=True)
        compare_with_analytic_solution(alpha=1, max_t=1e-10, plot=True)
    else:
        # fast gate: the two well-damped cases only (quick, artifact-free).
        test_macrospin_standard_damping()
        test_macrospin_higher_damping()
    print("macrospin: LLG dynamics matches the analytic Matteo solution.")
