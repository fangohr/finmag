import time
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt

L = 2 * np.pi
f_0 = 10
eps = 1e-6
ts = np.linspace(0, 2 * np.pi, 100)

def numpy_loop(t):
    """
    function depends on time and space, done in a for-loop

    """
    f = np.empty(mesh.num_vertices())
    for i, (x, y, z) in enumerate(mesh.coordinates()):
        f[i] = f_0
        if abs(x) > eps:
            f[i] *= np.sin(x)/x
        if abs(t) > eps:
            f[i] *= np.sin(t)/t
    return f

sinc_expr = df.Expression((
    "f_0"
    " * (fabs(x[0]) < eps ? 1 : sin(x[0])/x[0])"
    " * (fabs(t) < eps ? 1 : sin(t)/t)"), f_0 = f_0, t = 0.0, eps=eps)
def dolfin_against_numpy_loop(t):
    sinc_expr.t = t
    f = df.interpolate(sinc_expr, S)
    return f.vector().array()

def numpy_vectorised(t):
    """
    function depends only on time

    """
    f = np.empty(mesh.num_vertices())
    sint = np.sin(t)
    f.fill(sint)
    return f

simple_expr = df.Expression("sin(t)", t=0)
def dolfin_against_numpy_vectorised(t):
    simple_expr.t = t
    f = df.interpolate(simple_expr, S)
    return f.vector().array()

lines = ["b", "r", "b--", "r--"]
for i, method in enumerate([numpy_loop, dolfin_against_numpy_loop, numpy_vectorised, dolfin_against_numpy_vectorised]):
    vertices = []
    times = []
    for dL in [1, 2, 5, 7, 10, 12, 15, 17, 20]:
        mesh = df.Box(0, 0, 0, L, L, L, dL, dL, dL)
        S = df.FunctionSpace(mesh, "CG", 1)
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
