import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from math import sin
from simple_timer import SimpleTimer

benchmark = SimpleTimer()

"""
The goal is to compute the values of f(r, t) = sin(x) * sin(t) on a mesh for
different points in time. The cost of updating f a couple of times is measured.

"""

def time_dolfin(mesh, ts):
    """
    Uses a dolfin expression to compute the values of f on the mesh and the times in ts.

    """
    S = df.FunctionSpace(mesh, "CG", 1)
    expr = df.Expression("sin(x[0]) * sin(t)", t = 0.0)

    with benchmark:
        for t in ts:
            expr.t = t
            f = df.interpolate(expr, S)
    return benchmark.elapsed, f.vector().array()

def time_numpy_loop(mesh, ts):
    """
    Uses numpy and a loop over the mesh coordinates to compute the values of f
    on the mesh and the times in ts. This is what we think dolfin is doing
    in C++ internally when we call time_dolfin.

    """
    f = np.empty(mesh.num_vertices())
    S = df.FunctionSpace(mesh, "CG", 1)
    xs = df.interpolate(df.Expression("x[0]"), S).vector().array()

    with benchmark:
        for t in ts:
            for i, x in enumerate(xs):
                f[i] = sin(x) * sin(t)
    return benchmark.elapsed, f

def time_numpy_vectorised(mesh, ts):
    """
    Instead of looping over the coordinates like in time_numpy_loop, this
    uses vectorised numpy code.

    """
    f = np.empty(mesh.num_vertices())
    S = df.FunctionSpace(mesh, "CG", 1)
    xs = df.interpolate(df.Expression("x[0]"), S).vector().array()

    with benchmark:
        for t in ts:
            f[:] = np.sin(xs) * sin(t)
    return benchmark.elapsed, f

def time_numpy_smart(mesh, ts):
    """
    This method uses additional knowledge about the function at hand.

    The function `f(r, t) = sin(x) * sin(t)` is the product of the
    space-dependent part `sin(x)` and the time-dependent part `sin(t)`.

    Since the spatial discretisation doesn't change over time, the
    space-dependent part of the function only needs to be computed
    once. Multiplied by the time-dependent part at each time step, the full
    function is reconstructed.

    In a way, this method is not fair to the others, because it uses prior
    knowledge about the function which the computer can't derive on its own.

    """
    f = np.empty(mesh.num_vertices())
    S = df.FunctionSpace(mesh, "CG", 1)
    xs = df.interpolate(df.Expression("x[0]"), S).vector().array()
    f_space_dependent_part = np.sin(xs)

    with benchmark:
        for t in ts:
            f[:] = f_space_dependent_part * sin(t)
    return benchmark.elapsed, f

L = np.pi / 2
dLs = [1, 2, 5, 7, 10, 12, 17, 20]
ts = np.linspace(0, np.pi / 2, 100)
vertices = []
runtimes = []
alternate_methods = [time_numpy_loop, time_numpy_vectorised, time_numpy_smart]
for i, dL in enumerate(dLs):
    mesh = df.Box(0, 0, 0, L, L, L, dL, dL, dL)
    print "Running for a mesh with {} vertices [{}/{}].".format(
            mesh.num_vertices(), i+1, len(dLs))

    # reference
    t_dolfin, f_dolfin = time_dolfin(mesh, ts)

    # other methods
    runtimes_alternate_methods = []
    for method in alternate_methods:
        t, f = method(mesh, ts)
        assert np.max(np.abs(f - f_dolfin)) < 1e-14
        runtimes_alternate_methods.append(t)

    vertices.append(mesh.num_vertices())
    runtimes.append([t_dolfin] + runtimes_alternate_methods)

runtimes = zip(* runtimes)
plt.plot(vertices, runtimes[0], "b", label="dolfin interpolate expression")
plt.plot(vertices, runtimes[1], "r", label="numpy loop over coordinates")
plt.plot(vertices, runtimes[2], "g", label="numpy vectorised code")
plt.plot(vertices, runtimes[3], "c", label="numpy smart")
plt.xlabel("vertices")
plt.ylabel("time (s)")
plt.yscale("log")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), prop={'size':10})
plt.savefig("updating_expression.png")
print "Saved plot to 'updating_expression.png'."
