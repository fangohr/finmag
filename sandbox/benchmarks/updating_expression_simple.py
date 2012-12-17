import time
import math
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt

"""
function depends on time and space

Expression we use is sin(x[0])*sin(t). This is motivated by the sinc function, but 
we avoid any division by zero issues here.

"""

f_0 = 1
ts = np.linspace(0, 5, 100)

expr = df.Expression(
    "f_0 * sin(t) * sin(x[0])", f_0 = f_0, t = 0.0)

def dolfin_expression(t):
    expr.t = t
    f = df.interpolate(expr, S)
    # could creating the array here cost time? Get rid of it.
    #f.vector().array()
    return None

def node_loop(t):
    f = np.empty(mesh.num_vertices())
    for i, (x, y, z) in enumerate(mesh.coordinates()):
        f[i] = f_0 * math.sin(x) * math.sin(t)
    return None

def numpy(t):
    _ = f_0 * np.sin(xpos) * math.sin(t)
    return None 

lines = ["b-o", "r:x", "k^--"]
for i, method in enumerate([dolfin_expression, node_loop, numpy]):
    vertices = []
    times = []
    L = 10.
    for dL in [1, 2, 5, 7, 10, 12, 15, 17, 20]:
        mesh = df.Box(0, 0, 0, L, L, L, dL, dL, dL)
        coordinates = mesh.coordinates()
        xpos = coordinates[:,0]  # for numpy routine
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
