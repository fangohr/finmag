import dolfin
import numpy
import matplotlib.pyplot as plt

from finmag.sim.llg import LLG
from finmag.sim.tests.test_macrospin import make_analytical_solution
from scipy.integrate import odeint

mesh = dolfin.Box(0,1, 0,1, 0,1, 1,1,1)

alphas = numpy.linspace(0.01, 0.99, 1000)
max_deviations = []
mean_deviations = []

for alpha in alphas:
    llg = LLG(mesh)
    llg.alpha = alpha
    llg.initial_M((llg.MS, 0, 0))
    llg.H_app = (0, 0, 1e5)

    EXCHANGE = False
    llg.setup(EXCHANGE)

    ts = numpy.linspace(0, 1e-9, num=100)
    ys = odeint(llg.solve_for, llg.M, ts)

    M_analytical = make_analytical_solution(llg.MS, 1e5,
        alpha=llg.alpha, gamma=llg.gamma)

    deviations = []
    for i in range(len(ts)):

        M_computed = numpy.mean(ys[i].reshape((3, -1)), 1)
        M_ref = M_analytical(ts[i])

        # deviation as the average of the 3 components of the difference
        difference = numpy.abs(M_computed - M_ref)
        deviation = numpy.mean(difference)
        deviations.append(deviation)
    max_deviations.append(numpy.max(deviations))
    mean_deviations.append(numpy.mean(deviations))
    if numpy.max(deviations) > 0.3:
        print alpha, " max: ", numpy.max(deviations), " mean: ", numpy.mean(deviations)

plt.plot(alphas, max_deviations, "r.", label="max")
plt.plot(alphas, mean_deviations, "b.", label="mean")
plt.legend()
plt.title("deviation over alpha")
plt.ylabel("deviation")
plt.xlabel("alpha")
plt.savefig("deviation_over_alpha.pdf")
