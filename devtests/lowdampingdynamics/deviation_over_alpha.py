import dolfin
import numpy
import matplotlib.pyplot as plt

from finmag.sim.llg import LLG
from finmag.sim.tests.test_macrospin import make_analytical_solution
from scipy.integrate import odeint

"""
We gather the deviation between the analytical solution and the computed one
for some values of the tolerance of the time integrator and several values
for the damping term alpha.

"""

rtols_powers_of_ten = [-7, -8, -9, -10, -11] # easier LaTeX formatting
alphas = numpy.linspace(0.01, 0.99, 100)

mesh = dolfin.Box(0,1, 0,1, 0,1, 1,1,1)

mean_deviationss = []
for rtol_power_of_ten in rtols_powers_of_ten:
    rtol = pow(10, rtol_power_of_ten)
    print "#### New series for rtol={0}. ####".format(rtol)

    # One entry in this array corresponds to the average deviation between
    # the analytical solution and the computed solution for one value of alpha.
    mean_deviations = []
    for alpha in alphas:
        print "Solving for alpha={0}.".format(alpha)

        llg = LLG(mesh)
        llg.alpha = alpha
        llg.initial_M((llg.MS, 0, 0))
        llg.H_app = (0, 0, 1e5)
        llg.setup(False)

        M_analytical = make_analytical_solution(llg.MS, 1e5, llg.alpha, llg.gamma) 
    
        ts = numpy.linspace(0, 1e-9, num=100)
        ys = odeint(llg.solve_for, llg.M, ts, rtol=rtol)

        # One entry in this array corresponds to the deviation between the two
        # solutions for one particular moment during the simulation.
        deviations = []
        for i in range(len(ts)):
            M_computed = numpy.mean(ys[i].reshape((3, -1)), 1)
            M_ref = M_analytical(ts[i])
            # The difference of the two vectors has 3 components. The
            # deviation is the average over these components.
            deviations.append(numpy.mean(numpy.abs(M_computed - M_ref)))

        # This represents the addition of one point to the graph.
        mean_deviations.append(numpy.mean(deviations))

    # This represents one additional series in the graph.
    mean_deviationss.append(mean_deviations)

for i in range(len(rtols_powers_of_ten)):
    label = r"$rtol=1\cdot 10^{" + str(rtols_powers_of_ten[i]) + r"}$"
    plt.plot(alphas, mean_deviationss[i], ".", label=label)
plt.legend()
plt.title(r"Influence of $\alpha$ and rtol on the Deviation")
plt.ylabel("deviation")
plt.xlabel(r"$\alpha$")
plt.ylim((0, 0.1))
plt.savefig("deviation_over_alpha_rtols.pdf")

