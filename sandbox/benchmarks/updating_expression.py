import time
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt

"""
function depends on time and space

"""

L = 2 * np.pi
f_0 = 10
eps = 1e-6
ts = np.linspace(0, 2 * np.pi, 100)

expr = df.Expression((
    "f_0"
    " * (fabs(x[0]) < eps ? 1 : sin(x[0])/x[0])"
    " * (fabs(t) < eps ? 1 : sin(t)/t)"), f_0 = f_0, t = 0.0, eps=eps)
def dolfin_expression(t):
    expr.t = t
    f = df.interpolate(expr, S)
    return f.vector().array()

def numpy_loop(t):
    f = np.empty(mesh.num_vertices())
    for i, (x, y, z) in enumerate(mesh.coordinates()):
        f[i] = f_0
        if abs(x) > eps:
            f[i] *= np.sin(x)/x
        if abs(t) > eps:
            f[i] *= np.sin(t)/t
    return f

def numpy_vectorised_spatial(mesh):
    f = np.empty(mesh.num_vertices())
    for i, (x, y, z) in enumerate(mesh.coordinates()):
        f[i] = f_0
        if abs(x) > eps:
            f[i] *= np.sin(t)/x
    return f

def update_numpy_vectorised(t, f_t0):
    sint = np.sin(t)
    return f_t0 * sint

lines = ["b", "r", "r--"]
for i, method in enumerate([dolfin_expression, numpy_loop, update_numpy_vectorised]):
    vertices = []
    times = []
    for dL in [1, 2, 5, 7, 10, 12, 15, 17, 20]:
        mesh = df.Box(0, 0, 0, L, L, L, dL, dL, dL)
        S = df.FunctionSpace(mesh, "CG", 1)

        if method == update_numpy_vectorised:
            f_t0 = numpy_vectorised_spatial(mesh)
            method = lambda t: update_numpy_vectorised(t, f_t0)
            method.__name__ = "numpy_vectorised"

        vertices.append(mesh.num_vertices())
        start = time.time()
        for t in ts:
            method(t)
        runtime = time.time() - start
        times.append(runtime)
        print "{} ran for {:.2g} s.".format(method.__name__, runtime)
    plt.plot(vertices, times, lines[i], label=method.__name__)
plt.xlabel("vertices")
plt.ylabel("time (s)")
plt.yscale("log")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), prop={'size':10})
plt.savefig("updating_expression.png")
