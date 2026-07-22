# DOLFINx port (Task 30): converted from the legacy dolfin example.
# Changes vs legacy (all mechanically necessary for the ported package):
#   - print statements -> print() (Python 3).
#   - dolfin.BoxMesh(dolfin.Point(...)) -> dolfinx.mesh.create_box.
#   - make_analytic_solution imported from finmag.util.macrospin directly.
#   - INTERFACE-DRIFT: the legacy initial state
#     sim.llg._m_field.get_numpy_array_debug() (fed to odeint(sim.llg.solve_for,
#     ...)) was component-blocked; in the port get_numpy_array_debug() is raw
#     interleaved dof order while solve_for expects the component-blocked "xxx"
#     ordering. Use sim.llg.m_numpy (the xxx-ordered state) as the initial
#     condition. reshape((3, -1)) on solve_for's xxx output is unchanged.
#   - matplotlib plotting removed (deferred, Task 26).
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

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 2e-6

rtols_powers_of_ten = [-7, -8, -9, -10, -11]
mesh = dmesh.create_box(
    MPI.COMM_WORLD, [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    [1, 1, 1], dmesh.CellType.tetrahedron)


def test_deviations_over_alpha_and_tol(number_of_alphas=3):
    alphas = numpy.linspace(0.01, 1.00, number_of_alphas)

    for rtol_power_of_ten in rtols_powers_of_ten:
        rtol = pow(10, rtol_power_of_ten)
        print("#### New series for rtol={0}. ####".format(rtol))

        for alpha in alphas:
            print("Solving for alpha={0}.".format(alpha))

            sim = Simulation(mesh, 1, unit_length=1e-9)
            sim.alpha = alpha
            sim.set_m((1, 0, 0))
            sim.add(Zeeman((0, 0, 1e5)))

            ts = numpy.linspace(0, 1e-9, num=50)
            ys = odeint(sim.llg.solve_for, sim.llg.m_numpy, ts,
                        rtol=rtol, atol=rtol)

            M_analytical = make_analytic_solution(1e5, alpha, sim.gamma)
            for i in range(len(ts)):
                M_computed = numpy.mean(ys[i].reshape((3, -1)), 1)
                M_ref = M_analytical(ts[i])
                deviation = numpy.mean(numpy.abs(M_computed - M_ref))
                assert deviation < TOLERANCE


if __name__ == '__main__':
    n = 50 if os.environ.get("FINMAG_EXAMPLE_FULL") == "1" else 3
    test_deviations_over_alpha_and_tol(n)
    print("macrospin alpha/rtol: deviation from the analytic solution < 2e-6 "
          "across alpha and integrator tolerances.")
