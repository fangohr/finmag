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

def benchmark(demag, H_ref):
    REPETITIONS = 10
    start = time.time()
    for j in xrange(REPETITIONS):
        H = demag.compute_field()
    elapsed = (time.time() - start) / REPETITIONS

    H = H.reshape((3, -1))
    spread = abs(H[0].max() - H[0].min()) 
    error = abs(H.mean(axis=1)[0] - H_ref[0]) / abs(H_ref[0])

    return elapsed, spread, error

vertices = []
methods = ("FK default", "FK opt.", "FK LU", "GCR", "Treecode")
results = []

for maxh in maxhs:
    mesh = sphere(r=radius, maxh=maxh, directory="meshes")
    vertices.append(mesh.num_vertices())
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.Function(S3)
    m.assign(df.Constant(m_0))

    results_for_this_mesh = []
    for method in methods:
        if method == "FK default":
            parameters = {}
            parameters["phi_1_solver"] = "default"
            parameters["phi_1_preconditioner"] = "default"
            parameters["phi_2_solver"] = "default"
            parameters["phi_2_preconditioner"] = "default"
            demag = Demag("FK", solver_type="Krylov", parameters=parameters)
        elif method == "FK opt.":
            parameters = {}
            parameters["phi_1_solver"] = "cg"
            parameters["phi_1_preconditioner"] = "ilu"
            parameters["phi_2_solver"] = "cg"
            parameters["phi_2_preconditioner"] = "ilu"
            demag = Demag("FK", solver_type="Krylov", parameters=parameters)
        elif method == "FK LU":
            demag = Demag("FK", solver_type="LU")
        elif method == "GCR":
            demag = Demag("GCR")
        elif method == "Treecode":
            demag = Demag("Treecode")
        else:
            import sys
            print "What are you doing?"
            sys.exit(1)

        demag.setup(S3, m, Ms, unit_length)
        results_for_this_mesh.append(benchmark(demag, H_ref))
    results.append(results_for_this_mesh)

fig = plt.figure()

ax = fig.add_subplot(211)
for i, method in enumerate(methods):
    timings = [results_per_mesh[i][0] for results_per_mesh in results]
    ax.plot(vertices, timings, label=method)
ax.legend(loc="upper left", prop={'size': 8})
ax.set_xlabel("vertices")
ax.set_ylabel("time (s)")

ax = fig.add_subplot(212)
for i, method in enumerate(methods):
    errors = [results_per_mesh[i][2] for results_per_mesh in results]
    ax.plot(vertices, errors, label=method)
ax.legend(loc="upper right", prop={'size': 8})
ax.set_xlabel("vertices")
ax.set_ylabel("relative error (%)")

fig.tight_layout()
fig.savefig("benchmark.png")
