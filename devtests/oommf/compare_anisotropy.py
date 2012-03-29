import os
import dolfin as df
import numpy as np

from finmag.util.oommf.comparison import compare_anisotropy
from finmag.util.oommf import mesh
from finmag.sim.helpers import quiver, boxplot, stats

K1 = 45e4 # J/m^3

REL_TOLERANCE = 2e-1 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def test_small_problem():
    results = small_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_one_dimensional_problem():
    results = one_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def small_problem():
    # The oommf mesh corresponding to this problem only has a single cell.
    x_max = 1e-9; y_max = 1e-9; z_max = 1e-9;

    dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 5, 5, 5)
    oommf_mesh = mesh.Mesh((1, 1, 1), size=(x_max, y_max, z_max))

    def m_gen(coords):
        n = coords.shape[1]
        return np.array([np.ones(n), np.zeros(n), np.zeros(n)])

    return compare_anisotropy(m_gen, K1, (0, 0, 1),
            dolfin_mesh, oommf_mesh, name="small_problem")

def one_dimensional_problem():
    x_min = 0; x_max = 20e-9; x_n = 40;

    dolfin_mesh = df.Interval(x_n, x_min, x_max)
    oommf_mesh = mesh.Mesh((x_n, 1, 1), size=(x_max, 1e-12, 1e-12))

    def m_gen(coords):
        xs = coords[0]
        mx = 2 * xs/x_max - 1
        my = np.sqrt(1 - (2*xs/x_max - 1)*(2*xs/x_max - 1))
        mz = np.zeros(len(xs))
        return np.array([mx, my, mz])
    
    return compare_anisotropy(m_gen, K1, (0, 0, 1),
            dolfin_mesh, oommf_mesh, dims=1, name="1d")

def three_dimensional_problem():
    x_max = 20e-9; y_max = 10e-9; z_max = 10e-9;

    dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 20, 20)
    oommf_mesh = mesh.Mesh((40, 20, 20), size=(x_max, y_max, z_max))

    def m_gen(coords):
        xs = coords[0]
        mx = np.sin(1e9 * xs/3)**2
        my = np.zeros(len(xs))
        mz = np.ones(len(xs))
        return np.array([mx, my, mz])

    return compare_anisotropy(m_gen, K1, (0, 0, 1),
            dolfin_mesh, oommf_mesh, dims=1, name="1d")

if __name__ == '__main__':

    res0 = small_problem()
    print "1D problem, relative difference:\n", stats(res0["rel_diff"])
    res1 = one_dimensional_problem()
    print "1D problem, relative difference:\n", stats(res1["rel_diff"])
    res3 = three_dimensional_problem()
    print "3D problem, relative difference:\n", stats(res3["rel_diff"])
 
    # the "small" problem has only a single node to look at.
    # Because of this, we'll do an extra plot showing both anisotropy fields
    # at the same time.
    both = np.append(res0["oommf"], res0["foommf"])
    coords_once = np.array([r for r in res0["oommf_mesh"].iter_coords()])
    coords_twice = np.append(coords_once, coords_once, axis=0)
    quiver(both, coords_twice, filename=MODULE_DIR+"/single_anis_both.png")

    for res in [res0, res1, res3]: 
        prefix = MODULE_DIR + res["prob"] + "_anis_"
        quiver(res["m0"], res["mesh"], prefix+"m0.png")
        quiver(res["finmag"], res["mesh"], prefix+"finmag.png")
        quiver(res["oommf"], res["oommf_mesh"], prefix+"oommf.png")
        quiver(res["foommf"], res["oommf_mesh"], prefix+"foommf.png")
        quiver(res["rel_diff"], res["oommf_mesh"], prefix+"rel_diff.png")
        boxplot(res["rel_diff"], prefix+"rel_diff_box.png")
