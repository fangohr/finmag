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
    in C++ internally when we call time_dolfin. As expected, doing this in
    python is slower.

    """
    f = np.empty(mesh.num_vertices())

    with benchmark:
        for t in ts:
            for i, (x, y, z) in enumerate(mesh.coordinates()):
                f[i] = sin(x) * sin(t)
    return benchmark.elapsed, f

def time_numpy_no_loop(mesh, ts):
    """
    Instead of looping over the coordinates like in time_numpy_loop, this has
    saved the coordinates away so it can use vectorised numpy code.

    """
    f = np.empty(mesh.num_vertices())
    xs = mesh.coordinates()[:,0]

    with benchmark:
        for t in ts:
            f[:] = np.sin(xs) * sin(t)
    return benchmark.elapsed, f

def time_numpy_smart(mesh, ts):
    """
    This way of computing the function values is somewhat smarter than
    time_numpy_loop. The function is the product of a space-dependent and a
    time-dependent part. Since the spatial discretisation doesn't change over
    time, the space-dependent part of the function only needs to be computed
    once. Multiplied by the time-dependent part at each time step, the full
    function is reconstructed.

    In a way, this method is not fair to the others, because it uses prior
    knowledge about the function which the computer can't derive on its' own.

    """
    f = np.empty(mesh.num_vertices())
    f_spatial_only = np.sin(mesh.coordinates()[:,0])

    with benchmark:
        for t in ts:
            f[:] = f_spatial_only * sin(t)
    return benchmark.elapsed, f

L = np.pi / 2
dLs = [1, 2, 5, 7, 10, 12, 17, 20]
ts = np.linspace(0, np.pi / 2, 100)
vertices = []
runtimes = []
for i, dL in enumerate(dLs):
    mesh = df.Box(0, 0, 0, L, L, L, dL, dL, dL)
    print "Running for a mesh with {} vertices [{}/{}].".format(
            mesh.num_vertices(), i+1, len(dLs))

    t_dolfin, f_dolfin = time_dolfin(mesh, ts)
    t_numpy, f_numpy = time_numpy_loop(mesh, ts)
    t_Hans, f_Hans = time_numpy_no_loop(mesh, ts)
    t_smart, f_smart = time_numpy_smart(mesh, ts)

    vertices.append(mesh.num_vertices())
    runtimes.append((t_dolfin, t_numpy, t_Hans, t_smart))

    assert np.max(np.abs(f_dolfin - f_numpy)) < 1e-14
    assert np.max(np.abs(f_dolfin - f_Hans)) < 1e-14
    assert np.max(np.abs(f_dolfin - f_smart)) < 1e-14

runtimes = zip(* runtimes)
plt.plot(vertices, runtimes[0], "b", label="dolfin")
plt.plot(vertices, runtimes[1], "r", label="numpy loop")
plt.plot(vertices, runtimes[2], "g", label="numpy vectorised")
plt.plot(vertices, runtimes[3], "r--", label="numpy optimised")
plt.xlabel("vertices")
plt.ylabel("time (s)")
plt.yscale("log")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), prop={'size':10})
plt.savefig("updating_expression.png")
print "Saved plot to 'updating_expression.png'."
