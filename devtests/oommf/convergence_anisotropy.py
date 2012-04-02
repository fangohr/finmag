import os
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from finmag.util.oommf.comparison import compare_anisotropy
from finmag.util.oommf import mesh
from finmag.sim.helpers import quiver

K1 = 45e4 # J/m^3

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

rel_diff_maxs = []
rel_diff_means = []
vertices = []

for x_n in [4, 8, 20, 40, 80]:
  x_max = 20e-9; y_max = x_max/2; z_max = x_max/4; # proportions 4 * 2 * 1

  dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, x_n, x_n/2, x_n/4)
  oommf_mesh = mesh.Mesh((x_n, x_n/2, x_n/4), size=(x_max, y_max, z_max))

  def m_gen(coords):
      n = coords.shape[1]
      return np.array([np.ones(n), np.zeros(n), np.zeros(n)])

  res = compare_anisotropy(m_gen, K1, (1, 0, 0),
          dolfin_mesh, oommf_mesh, name="small_problem")

  vertices.append(dolfin_mesh.num_vertices())  
  rel_diff_maxs.append(np.max(res["rel_diff"]))
  rel_diff_means.append(np.mean(res["rel_diff"]))

print vertices
print rel_diff_maxs
print rel_diff_means
plt.plot(vertices, rel_diff_means)
plt.savefig(MODULE_DIR + "anis_convergence.png")
