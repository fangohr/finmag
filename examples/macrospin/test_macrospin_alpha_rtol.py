import py
import os
import dolfin
import numpy
import logging
import matplotlib.pyplot as plt
from finmag import Simulation
from finmag.energies import Zeeman
from test_macrospin import make_analytic_solution
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

rtols_powers_of_ten = [-7, -8, -9, -10, -11] # easier LaTeX formatting
mesh = dolfin.BoxMesh(0,1, 0,1, 0,1, 1,1,1)

def test_deviations_over_alpha_and_tol(number_of_alphas=5, do_plot=False):
    alphas = numpy.linspace(0.01, 1.00, number_of_alphas)

    max_deviationss = []
    for rtol_power_of_ten in rtols_powers_of_ten:
        rtol = pow(10, rtol_power_of_ten)
        print "#### New series for rtol={0}. ####".format(rtol)

        # One entry in this array corresponds to the maximum deviation between
        # the analytical solution and the computed solution for one value of alpha.
        max_deviations = []
        for alpha in alphas:
            print "Solving for alpha={0}.".format(alpha)

            sim = Simulation(mesh, 1)
            sim.alpha = alpha
            sim.set_m((1, 0, 0))
            sim.add(Zeeman((0, 0, 1e5)))

            ts = numpy.linspace(0, 1e-9, num=50)
            ys = odeint(sim.llg.solve_for, sim.llg.m, ts, rtol=rtol, atol=rtol)

            # One entry in this array corresponds to the deviation between the two
            # solutions for one particular moment during the simulation.
            deviations = []
            M_analytical = make_analytic_solution(1e5, alpha, sim.gamma)
            for i in range(len(ts)):
                M_computed = numpy.mean(ys[i].reshape((3, -1)), 1)
                M_ref = M_analytical(ts[i])
                # The difference of the two vectors has 3 components. The
                # deviation is the average over these components.
                deviation = numpy.mean(numpy.abs(M_computed - M_ref))
                assert deviation < TOLERANCE
                deviations.append(deviation)

            # This represents the addition of one point to the graph.
            max_deviations.append(numpy.max(deviations))

        # This represents one additional series in the graph.
        max_deviationss.append(max_deviations)

    if do_plot:
        for i in range(len(rtols_powers_of_ten)):
            label = r"$rtol=1\cdot 10^{" + str(rtols_powers_of_ten[i]) + r"}$"
            plt.plot(alphas, max_deviationss[i], ".", label=label)
        plt.legend()
        plt.title(r"Influence of $\alpha$ and rtol on the Deviation")
        plt.ylabel("deviation")
        plt.xlabel(r"$\alpha$")
        plt.ylim((0, 1e-6))
        plt.savefig(os.path.join(MODULE_DIR, "deviation_over_alpha_rtols.pdf"))

if __name__ == '__main__':
    test_deviations_over_alpha_and_tol(50, do_plot=True)
