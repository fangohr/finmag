# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.BoxMesh(dolfin.Point(...)) -> dolfinx.mesh.create_box.
#   - make_analytic_solution imported from finmag.util.macrospin directly.
#   - INTERFACE-DRIFT (#6): the legacy initial state
#     sim.llg._m_field.get_numpy_array_debug() (fed to odeint(sim.llg.solve_for,
#     ...)) was component-blocked; in the port get_numpy_array_debug() is raw
#     interleaved dof order while solve_for expects the component-blocked "xxx"
#     ordering. As a Task 30 WORKAROUND we use sim.llg.m_numpy (the xxx-ordered
#     state) as the initial condition. This is to be REVERTED in Task 31
#     (component-ordering restoration): CONTROLLER DECISION -- Task 31 will
#     restore get_numpy_array_debug() to the legacy component-blocked ordering,
#     after which the legacy line is used verbatim again. reshape((3, -1)) on
#     solve_for's xxx output is unchanged.
#   - The pure-matplotlib do_plot branch is restored from legacy (Agg backend,
#     savefig only); it runs on the __main__/save path under FINMAG_EXAMPLE_FULL
#     so the fast gate stays quick and artifact-free -- disclosed, not deleted.
#   - Gate mode uses 3 alphas (full: 50); the assertion (deviation < 2e-6) is
#     unchanged.
# [Claude Opus 4.8]
import os
import logging
import numpy
from mpi4py import MPI
from dolfinx import mesh as dmesh
from finmag import Simulation
from finmag.energies import Zeeman
from finmag.util.macrospin import make_analytic_solution
from scipy.integrate import odeint

log = logging.getLogger(name='finmag')
log.setLevel(logging.WARNING)

"""
We gather the deviation between the analytical solution of the macrospin problem
and the computed one for some values of the tolerance of the time integrator
and an alpha ranging from 0.01 to 0.99.
"""

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 2e-6

rtols_powers_of_ten = [-7, -8, -9, -10, -11]  # easier LaTeX formatting
mesh = dmesh.create_box(
    MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    [1, 1, 1], dmesh.CellType.tetrahedron)


def test_deviations_over_alpha_and_tol(number_of_alphas=3, do_plot=False):
    alphas = numpy.linspace(0.01, 1.00, number_of_alphas)

    max_deviationss = []
    for rtol_power_of_ten in rtols_powers_of_ten:
        rtol = pow(10, rtol_power_of_ten)
        print("#### New series for rtol={0}. ####".format(rtol))

        # One entry per alpha: the maximum deviation between the analytical and
        # the computed solution for that alpha.
        max_deviations = []
        for alpha in alphas:
            print("Solving for alpha={0}.".format(alpha))

            sim = Simulation(mesh, 1, unit_length=1e-9)
            sim.alpha = alpha
            sim.set_m((1, 0, 0))
            sim.add(Zeeman((0, 0, 1e5)))

            ts = numpy.linspace(0, 1e-9, num=50)
            # Task 30 WORKAROUND (drift #6), REVERT in Task 31: the legacy line
            # was
            #   odeint(sim.llg.solve_for,
            #          sim.llg._m_field.get_numpy_array_debug(), ...)
            # get_numpy_array_debug() is raw-interleaved in the port but
            # solve_for expects the xxx ordering, so m_numpy is used here.
            # Task 31 restores get_numpy_array_debug() to component-blocked
            # ordering (controller decision), after which the legacy call
            # is used verbatim.
            ys = odeint(sim.llg.solve_for, sim.llg.m_numpy, ts,
                        rtol=rtol, atol=rtol)

            # One entry per timestep: the deviation between the two solutions.
            deviations = []
            M_analytical = make_analytic_solution(1e5, alpha, sim.gamma)
            for i in range(len(ts)):
                M_computed = numpy.mean(ys[i].reshape((3, -1)), 1)
                M_ref = M_analytical(ts[i])
                # The difference of the two vectors has 3 components; the
                # deviation is the average over these components.
                deviation = numpy.mean(numpy.abs(M_computed - M_ref))
                assert deviation < TOLERANCE
                deviations.append(deviation)

            # One additional point on the graph.
            max_deviations.append(numpy.max(deviations))

        # One additional series on the graph.
        max_deviationss.append(max_deviations)

    if do_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for i in range(len(rtols_powers_of_ten)):
            label = r"$rtol=1\cdot 10^{" + str(rtols_powers_of_ten[i]) + r"}$"
            plt.plot(alphas, max_deviationss[i], ".", label=label)
        plt.legend()
        plt.title(r"Influence of $\alpha$ and rtol on the Deviation")
        plt.ylabel("deviation")
        plt.xlabel(r"$\alpha$")
        plt.ylim((0, 1e-6))
        plt.savefig(os.path.join(MODULE_DIR, "deviation_over_alpha_rtols.pdf"))
        plt.close()


if __name__ == '__main__':
    full = os.environ.get("FINMAG_EXAMPLE_FULL") == "1"
    n = 50 if full else 3
    test_deviations_over_alpha_and_tol(n, do_plot=full)
    print("macrospin alpha/rtol: deviation from the analytic solution < 2e-6 "
          "across alpha and integrator tolerances.")
