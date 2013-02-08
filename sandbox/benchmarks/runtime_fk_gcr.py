import dolfin as df
from finmag.energies import Demag
from finmag.util.meshes import sphere
from simple_timer import SimpleTimer
import matplotlib.pyplot as plt

benchmark = SimpleTimer()
maxhs = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
solvers = ["FK", "GCR"]
timings = [[], []]
errors = [[], []]

for maxh in maxhs:
    mesh = sphere(r = 2.0, maxh = maxh, save_result=True)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.Function(S3)
    m.assign(df.Constant((1, 0, 0)))

    for i, solver in enumerate(solvers):
        demag = Demag(solver)
        demag.setup(S3, m, Ms=1, unit_length=1e-9)
        with benchmark:
            for j in range(10):
                demag.compute_field()
        field = demag.compute_field().reshape((3, -1)).mean(axis=1)
        print field
        error = abs(field[0] + 1.0/3)
        print error
        timings[i].append(benchmark.elapsed)
        errors[i].append(error)

plt.plot(maxhs, timings[0], label=solvers[0])
plt.plot(maxhs, timings[1], label=solvers[1])
plt.legend()
plt.xlabel("maxh (nm)")
plt.ylabel("10x compute field (s)")
plt.savefig("timings_fk_gcr.png")
plt.clf()

plt.plot(maxhs, errors[0], label=solvers[0])
plt.plot(maxhs, errors[1], label=solvers[1])
plt.legend()
plt.xlabel("maxh (nm)")
plt.ylabel("error in m_x")
plt.savefig("error_fk_gcr.png")
