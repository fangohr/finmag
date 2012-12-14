import time
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from finmag.energies import TimeZeeman

TOL = 1e-10

class SimpleTimeZeeman(object):
    """
    Works similar to TimeZeeman, except it generates the field using
    a python function and not a dolfin expression.

    """
    def __init__(self, ignore_this_param):
        self.in_jacobian = False

    def setup(self, S3, m, Ms, unit_length=1):
        self.mesh_coordinates = S3.mesh().coordinates()
        self.H = np.zeros((3, self.mesh_coordinates.shape[0]))

    def update(self, t):
        self.t = t
        self.H[2] = H0 * np.sin(t)

    def compute_field(self):
        return self.H.ravel()

H0 = 1
H_ext_expr = df.Expression(("0.0", "0.0", "H0 * sin(t)"), H0=H0, t=0.0)
steps = 10
ts = np.linspace(0, 1*np.pi, steps)
H_ref = H0 * np.sin(ts)

for Zee in [SimpleTimeZeeman, TimeZeeman]:
    print "Running with {}.".format(Zee.__name__)
    vertices = []
    times = []
    for div in [1, 10, 20, 50]:
        mesh = df.Box(0, 0, 0, 100, 100, 100, div, div, div)
        vertices.append(mesh.num_vertices())
        print "\t mesh with {} vertices...".format(mesh.num_vertices())
        S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
        m = df.Function(S3)
        Ms = 1
        unit_length = 1e-9

        H = Zee(H_ext_expr)
        H.setup(S3, m, Ms, unit_length)

        start = time.time()
        for i, t in enumerate(ts):
            print "\t\t step {}/{}...".format(i+1, steps)
            H.update(t)
            field = H.compute_field().reshape((3, -1)).mean(1)
            assert abs(field[2] - H_ref[i]) < TOL
        stop = time.time()
        times.append(stop - start)
    plt.plot(vertices, times, label=Zee.__name__)
plt.xlabel("vertices")
plt.ylabel("time (s)")
plt.yscale("log")
plt.legend()
plt.title("cost of updating the time dependent external field")
plt.savefig("updating_Zeeman.png")
