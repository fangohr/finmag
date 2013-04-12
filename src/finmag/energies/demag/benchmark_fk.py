import time
import pickle
import numpy as np
import dolfin as df
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from finmag.energies import Demag
from finmag.util.meshes import box
from finmag.native.llg import compute_bem_fk

now = time.time
create_mesh = lambda maxh: box(0, 0, 0, 500, 25, 1, maxh, directory="meshes")
maxhs = np.arange(1.0, 3.7, 0.2)
Ms = 1
m_0 = (1, 0, 0)
unit_length = 1e-9
default_params = ("default", "default", "default", "default")
opt_params = ("cg", "ilu", "cg", "ilu")
repetitions = 10
results_file = "results_fk_benchmark.txt"


def run_demag(repetitions, params, S3, m, Ms, unit_length, bem=None, b2g_map=None):
    demag = Demag("FK")
    demag.parameters["phi_1_solver"] = params[0]
    demag.parameters["phi_1_preconditioner"] = params[1]
    demag.parameters["phi_2_solver"] = params[2]
    demag.parameters["phi_2_preconditioner"] = params[3]
    if bem is not None:
        demag.precomputed_bem(bem, b2g_map)
    demag.setup(S3, m, Ms, unit_length)
    start = now()
    for j in xrange(repetitions):
        H = demag.compute_field()
    runtime = (now() - start) / repetitions
    return H, runtime

try:
    results = np.loadtxt(results_file)
except IOError:
    results = np.zeros((len(maxhs), 4))
    for i, maxh in enumerate(maxhs):
        print "Mesh {}/{} with maxh = {:.3}.".format(i + 1, len(maxhs), maxh)
        mesh = create_mesh(maxh)
        S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
        m = df.Function(S3)
        m.assign(df.Constant(m_0))

        # Pre-compute BEM to save time.
        bem, b2g_map = compute_bem_fk(df.BoundaryMesh(mesh, 'exterior', False))

        print "Computing demagnetising field with default solver parameters..."
        H_default, runtime_default = run_demag(
            repetitions, default_params, S3, m, Ms, unit_length, bem, b2g_map)

        print "Computing demagnetising field with optimised solver parameters..."
        H_opt, runtime_opt = run_demag(
            repetitions, opt_params, S3, m, Ms, unit_length, bem, b2g_map)

        results[i, 0] = mesh.num_vertices()
        results[i, 1] = runtime_default
        results[i, 2] = runtime_opt
        results[i, 3] = np.max(np.abs(H_default - H_opt))
        np.savetxt(results_file, results)  # Save results after every step.

vertices = results[:,0]
runtimes_default = results[:,1]
runtimes_opt = results[:,2]
deviation = results[:,3]

fig = plt.figure()
ax = fig.add_subplot(211)
ax.set_title("Runtime Comparison")
ax.plot(vertices, runtimes_default, label="default FK settings")
ax.plot(vertices, runtimes_opt, label="optimised FK settings")
ax.legend(loc=2)
ax.set_xlabel("vertices")
ax.set_ylabel("time (s)")

ax = fig.add_subplot(212)
ax.set_title("Deviation from Solution Obtained with Default Settings")
ax.plot(vertices, deviation)
ax.set_xlabel("vertices")
ax.set_ylabel("max. deviation")

fig.tight_layout()
fig.savefig("results_fk_benchmark.png")
