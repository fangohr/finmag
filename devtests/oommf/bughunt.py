import os
import dolfin as df
import numpy as np

from finmag.util.oommf.comparison import compare_anisotropy, oommf_m0
from finmag.util.oommf import mesh
from finmag.sim.helpers import quiver

K1 = 45e4 # J/m^3

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

#The oommf mesh corresponding to this problem only has a single cell.
x_max = 1e-9; y_max = 1e-9; z_max = 1e-9;

dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 5, 5, 5)
oommf_mesh = mesh.Mesh((1, 1, 1), size=(x_max, y_max, z_max))

def m_gen(coords):
    n = coords.shape[1]
    return np.array([np.ones(n), np.zeros(n), np.zeros(n)])

m0 = oommf_m0(m_gen, oommf_mesh)
print m0.flat

coords = np.array(zip(* dolfin_mesh.coordinates()))
m0_f = m_gen(coords).flatten()
print m0_f

sqrt3 = 1 / np.sqrt(3)
res = compare_anisotropy(m_gen, K1, (sqrt3, sqrt3, sqrt3),
        dolfin_mesh, oommf_mesh, name="small_problem")


# the "small" problem has only a single node to look at.
# Because of this, we'll do an extra plot showing both anisotropy fields
# at the same time.
both = np.append(res["oommf_anis"], res["anis"])
coords_once = np.array([r for r in res["oommf_mesh"].iter_coords()])
coords_twice = np.append(coords_once, coords_once, axis=0)
quiver(both, coords_twice, filename=MODULE_DIR+"/single_anis_both.png")
