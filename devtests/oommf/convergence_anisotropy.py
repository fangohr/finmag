import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from finmag.util.oommf.comparison import compare_anisotropy
from finmag.util.oommf import mesh
from finmag.sim.helpers import quiver

K1 = 45e4 # J/m^3

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

max_rdiffs = [[],[]]
vertices = [[],[]]

for x_n in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    x_min = 0; x_max = 100e-9;
    dolfin_mesh = df.Interval(int(x_n), x_min, x_max)
    oommf_mesh = mesh.Mesh((20, 1, 1), size=(x_max, 1e-12, 1e-12))

    def m_gen(rs):
      xs = rs[0]
      return np.array([2*xs/x_max - 1, np.sqrt(1 - (2*xs/x_max - 1)**2), np.zeros(len(xs))])

    res = compare_anisotropy(m_gen, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, dims=1, name="1D")

    vertices[0].append(dolfin_mesh.num_vertices())  
    max_rdiffs[0].append(np.max(res["rel_diff"]))

for x_n in [10, 80, 350]:
    x_max = 100e-9; y_max = z_max = x_max/10;
    y_n = z_n = x_n/10;
    dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, x_n, y_n, z_n)
    oommf_mesh = mesh.Mesh((50, 5, 5), size=(x_max, y_max, z_max))

    def m_gen(rs):
      xs = rs[0]
      return np.array([xs/x_max, np.sqrt(1 - (xs/x_max)**2), np.zeros(len(xs))])

    res = compare_anisotropy(m_gen, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, dims=3, name="3D")
 
    vertices[1].append(dolfin_mesh.num_vertices())  
    max_rdiffs[1].append(np.max(res["rel_diff"]))

print vertices
print max_rdiffs

plt.xlabel("vertices")
plt.ylabel("rel. max diff.")
plt.loglog(vertices[0], max_rdiffs[0], "--")
plt.loglog(vertices[0], max_rdiffs[0], "o", label="1d problem")
plt.loglog(vertices[1], max_rdiffs[1], "--")
plt.loglog(vertices[1], max_rdiffs[1], "o", label="3d problem")
plt.legend()
plt.savefig(MODULE_DIR + "anis_convergence.png")
plt.show()
