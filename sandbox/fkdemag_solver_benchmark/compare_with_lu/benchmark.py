import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import numpy as np
import dolfin as df
from finmag.energies import Demag
from finmag.util.meshes import sphere

radius = 5.0
maxhs = [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
unit_length = 1e-9
m_0 = (1, 0, 0)
Ms = 1
H_ref = np.array((- Ms / 3.0, 0, 0))

vertices = []
solvers_label = ["FK", "FK opt.", "GCR", "Treecode", "FK LU"]
timings = [[], [], [], [], []]
errors = [[], [], [], [], []]

for maxh in maxhs:
    mesh = sphere(r=radius, maxh=maxh, directory="meshes")
    vertices.append(mesh.num_vertices())
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.Function(S3)
    m.assign(df.Constant(m_0))

    for i in xrange(5):
        if i == 0:
            parameters = {}
            parameters["phi_1_solver"] = "default"
            parameters["phi_1_preconditioner"] = "default"
            parameters["phi_2_solver"] = "default"
            parameters["phi_2_preconditioner"] = "default"
            solver_type = "Krylov" 
            demag = Demag("FK", solver_type=solver_type, parameters=parameters)
        elif i == 1:
            parameters = {}
            parameters["phi_1_solver"] = "cg"
            parameters["phi_1_preconditioner"] = "ilu"
            parameters["phi_2_solver"] = "cg"
            parameters["phi_2_preconditioner"] = "ilu"
            solver_type = "Krylov" 
            demag = Demag("FK", solver_type=solver_type, parameters=parameters)
        elif i == 2:
            solver = "GCR"
            demag = Demag(solver)
        elif i == 3:
            solver = "Treecode"
            demag = Demag(solver)
        elif i == 4:
            solver = "FK"
            demag = Demag(solver, solver_type="LU")
        demag.setup(S3, m, Ms, unit_length)

        start = time.time()
        for j in xrange(10):
            H = demag.compute_field()
        elapsed = (time.time() - start) / 10.0

        H = H.reshape((3, -1)).mean(axis=1)
        error = abs(H[0] - H_ref[0]) / abs(H_ref[0])
        timings[i].append(elapsed)
        errors[i].append(error)

print solvers_label, timings, errors

fig = plt.figure()
ax = fig.add_subplot(211)
for i, label in enumerate(solvers_label):
    ax.plot(vertices, timings[i], label=label)
ax.legend(loc=2)
ax.set_xlabel("vertices")
ax.set_ylabel("time (s)")

ax = fig.add_subplot(212)
for i, label in enumerate(solvers_label):
    ax.plot(vertices, errors[i], label=label)
ax.legend(loc=2)
ax.set_xlabel("vertices")
ax.set_ylabel("relative error (%)")

fig.tight_layout()
fig.savefig("benchmark.png")
