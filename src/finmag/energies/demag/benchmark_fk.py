import time
import numpy as np
import dolfin as df
from finmag.energies import Demag
from finmag.util.meshes import box
import matplotlib.pyplot as plt

maxhs = [1.5, 2.0, 3.0, 3.5]
unit_length = 1e-9
m_0 = (1, 0, 0)
Ms = 1

vertices = []
solvers = ["FK", "FK"]
solvers_label = ["FK", "FK opt"]
timings = [[], []]
errors = [[], []]

for maxh in maxhs:
    mesh = box(0, 0, 0, 500, 25, 1, maxh=maxh, directory="meshes")
    vertices.append(mesh.num_vertices())
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.Function(S3)
    m.assign(df.Constant(m_0))

    H_ref = None

    for i, solver in enumerate(solvers):
        demag = Demag(solver)
        if i == 0:
            demag.parameters["phi_1_solver"] = "default"
            demag.parameters["phi_1_preconditioner"] = "default"
            demag.parameters["phi_2_solver"] = "default"
            demag.parameters["phi_2_preconditioner"] = "default"
        if i == 1:
            demag.parameters["phi_1_solver"] = "cg"
            demag.parameters["phi_1_preconditioner"] = "ilu"
            demag.parameters["phi_2_solver"] = "cg"
            demag.parameters["phi_2_preconditioner"] = "ilu"

        demag.setup(S3, m, Ms, unit_length)

        start = time.time()
        for j in xrange(10):
            H = demag.compute_field()
        elapsed = (time.time() - start) / 10.0

        if H_ref is None:
            H_ref = H
        error = np.max(np.abs(H - H_ref))
        timings[i].append(elapsed)
        errors[i].append(error)

fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("Runtime")
for i, solver in enumerate(solvers):
    ax.plot(vertices, timings[i], label=solvers[i])
ax.legend(loc=2)
ax.set_xlabel("vertices")
ax.set_ylabel("time (s)")

ax = fig.add_subplot(212)
ax.set_title("Deviation from Default Parameters")
for i, solver in enumerate(solvers):
    ax.plot(vertices, errors[i], label=solvers[i])
ax.legend(loc=2)
ax.set_xlabel("vertices")
ax.set_ylabel("deviation")

fig.tight_layout()
fig.savefig("benchmark_fk.png")
