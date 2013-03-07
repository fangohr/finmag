import os
import numpy as np
import dolfin as df
import matplotlib.pyplot as plt
from finmag.util.helpers import stats 
from finmag.util.oommf import mesh
from finmag.util.oommf.comparison import compare_exchange 

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
Ms = 8.6e6
A = 1.3e-11

def test_one_dimensional_problem():
    REL_TOLERANCE = 1e-3
    results = one_dimensional_problem(5000)
    assert np.nanmax(np.mean(results["rel_diff"], axis=1)) < REL_TOLERANCE

def one_dimensional_problem(vertices):
    x_min = 0; x_max = 100e-9;

    dolfin_mesh = df.IntervalMesh(vertices, x_min, x_max)
    oommf_mesh = mesh.Mesh((vertices, 1, 1), size=(x_max, 1e-12, 1e-12))

    def m_gen(coords):
        xs = coords[0]
        return np.array([np.sqrt(xs/x_max), np.sqrt(1 - xs/x_max), np.zeros(len(xs))])

    return compare_exchange(m_gen, Ms, A, dolfin_mesh, oommf_mesh, dims=1, name="1d")

if __name__ == '__main__':

    vertices = []
    mean_diffs = []

    for n in [10, 1e2, 1e3, 1e4, 1e5]:
      res = one_dimensional_problem(int(n))
      print "1D problem ({} nodes) relative difference:".format(n)
      print stats(res["rel_diff"])
      vertices.append(int(n))
      mean_diffs.append(np.nanmax(np.mean(res["rel_diff"], axis=1)))

    plt.xlabel("vertices")
    plt.ylabel("rel. mean diff.")
    plt.loglog(vertices, mean_diffs, "--")
    plt.loglog(vertices, mean_diffs, "o", label="1d problem")
    plt.legend()
    plt.savefig(os.path.join(MODULE_DIR, "exchange_convergence.png"))
