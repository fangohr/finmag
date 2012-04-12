import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from finmag.util.oommf.comparison import compare_anisotropy
from finmag.util.oommf import mesh

K1 = 45e4 # J/m^3

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

max_rdiffs = [[], []]
mean_rdiffs = [[], []]
vertices = [[], []]

print "1D problem..."

def m_gen(rs):
    xs = rs[0]
    return np.array([xs/x_max, np.sqrt(1 - (xs/x_max)**2), np.zeros(len(xs))])

for x_n in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:
    x_min = 0; x_max = 100e-9;
    dolfin_mesh = df.Interval(int(x_n), x_min, x_max)
    oommf_mesh = mesh.Mesh((int(x_n), 1, 1), size=(x_max, 1e-12, 1e-12))
    res = compare_anisotropy(m_gen, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, dims=1, name="1D")

    vertices[0].append(dolfin_mesh.num_vertices())  
    mean_rdiffs[0].append(np.mean(res["rel_diff"]))
    max_rdiffs[0].append(np.max(res["rel_diff"]))

print "3D problem..."

for x_n in [10, 100, 200, 300]:
    x_max = 100e-9; y_max = z_max = 1e-9;
    y_n = z_n = x_n/10;
    dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, x_n, y_n, z_n)
    oommf_mesh = mesh.Mesh((x_n, y_n, z_n), size=(x_max, y_max, z_max))
    res = compare_anisotropy(m_gen, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, dims=3, name="3D")
 
    vertices[1].append(dolfin_mesh.num_vertices())  
    mean_rdiffs[1].append(np.mean(res["rel_diff"]))
    max_rdiffs[1].append(np.max(res["rel_diff"]))

plt.xlabel("number of vertices")
plt.ylabel("relative difference")
plt.loglog(vertices[0], mean_rdiffs[0], "b--")
plt.loglog(vertices[0], mean_rdiffs[0], "bo", label="1d mean")
plt.loglog(vertices[1], mean_rdiffs[1], "r--")
plt.loglog(vertices[1], mean_rdiffs[1], "ro", label="3d mean")
#plt.loglog(vertices[0], max_rdiffs[0], "b-.")
#plt.loglog(vertices[0], max_rdiffs[0], "bs", label="1d max")
#plt.loglog(vertices[1], max_rdiffs[1], "r-.")
#plt.loglog(vertices[1], max_rdiffs[1], "rs", label="3d max")
plt.legend()
plt.savefig(MODULE_DIR + "anis_convergence.png")
plt.show()
