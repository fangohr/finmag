import os
import dolfin
import numpy
import matplotlib.pyplot as plt

from finmag.sim.llg import LLG
from finmag.energies import Zeeman
from test_macrospin import make_analytic_solution
from scipy.integrate import odeint

"""
We gather the deviation between the analytical solution of the macrospin problem
and the computed one for some values of the tolerance of the time integrator
and an alpha ranging from 0.01 to 0.99.

"""

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 2e-6

rtols_powers_of_ten = [-7, -8, -9, -10, -11] # easier LaTeX formatting
mesh = dolfin.Box(0,1, 0,1, 0,1, 1,1,1)

def test_deviations_over_alpha_and_tol(number_of_alphas=5, do_plot=False):
    alphas = numpy.linspace(0.01, 0.99, number_of_alphas)

    max_deviationss = []
    for rtol_power_of_ten in rtols_powers_of_ten:
        rtol = pow(10, rtol_power_of_ten)
        print "#### New series for rtol={0}. ####".format(rtol)

        # One entry in this array corresponds to the maximum deviation between
        # the analytical solution and the computed solution for one value of alpha.
        max_deviations = []
        for alpha in alphas:
            print "Solving for alpha={0}.".format(alpha)

            S1 = dolfin.FunctionSpace(mesh, "Lagrange", 1)
            S3 = dolfin.VectorFunctionSpace(mesh, "Lagrange", 1)
            llg = LLG(S1, S3)
            llg.alpha = alpha
            llg.set_m((1, 0, 0))

            H_app = Zeeman((0, 0, 1e5))
            H_app.setup(S3, llg._m, Ms=1)
            llg.interactions.append(H_app)

            M_analytical = make_analytic_solution(1e5, llg.alpha, llg.gamma) 
        
            ts = numpy.linspace(0, 1e-9, num=50)
            ys = odeint(llg.solve_for, llg.m, ts, rtol=rtol, atol=rtol)

            # One entry in this array corresponds to the deviation between the two
            # solutions for one particular moment during the simulation.
            deviations = []
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
        plt.savefig(MODULE_DIR+"/deviation_over_alpha_rtols.pdf")

if __name__ == '__main__':
    test_deviations_over_alpha_and_tol(100, do_plot=True)
