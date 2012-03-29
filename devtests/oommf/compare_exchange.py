import os
import dolfin as df
import numpy as np
import finmag.sim.helpers as h
from finmag.util.oommf import mesh
from finmag.util.oommf.comparison import compare_exchange 

REL_TOLERANCE = 35 # goal: < 1e-3
MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def test_one_dimensional_problem():
    results = one_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_three_dimensional_problem():
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def one_dimensional_problem():
    x_min = 0; x_max = 20e-9; x_n = 40;

    dolfin_mesh = df.Interval(x_n, x_min, x_max)
    oommf_mesh = mesh.Mesh((x_n, 1, 1), size=(x_max, 1e-12, 1e-12))

    def m_gen(coords):
        xs = coords[0]
        mx = np.sqrt(xs/x_max)
        # When x=x_max (at the last node), x/x_max = 1 and sin(4*PI)
        # should equal exactly 0. Of course, instead of an exact zero, we get
        # an ever so small positive value, which lets the argument of the
        # square root turn negative. To prevent this, we chose 0 at that point.
        my_squared = 1 - xs/x_max - (np.sin(4 * np.pi * xs/x_max)/10)**2
        my = np.sqrt(np.maximum(my_squared, np.zeros(len(my_squared))))
        mz = np.sin(4 * np.pi * xs/x_max) / 10
        return np.array([mx, my, mz])

    return compare_exchange(m_gen, dolfin_mesh, oommf_mesh, dims=1, name="1d")

def three_dimensional_problem():
    x_max = 20e-9; y_max = 10e-9; z_max = 10e-9;

    dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 40, 10, 10)
    oommf_mesh = mesh.Mesh((20, 10, 10), size=(x_max, y_max, z_max))

    def m_gen(rs):
        n = rs.shape[1]
        return np.array([np.sin(1e9 * rs[0] / 3)**2, np.zeros(n), np.ones(n)])

    return compare_exchange(m_gen, dolfin_mesh, oommf_mesh, name="3d")

if __name__ == '__main__':
    res1 = one_dimensional_problem()
    print "1D problem, relative difference:\n", h.stats(res1["rel_diff"])
    res3 = three_dimensional_problem()
    print "3D problem, relative difference:\n", h.stats(res3["rel_diff"])

    for res in [res1, res3]: 
        prefix = MODULE_DIR + res["name"] + "_exc_"
        # images are worthless if the colormap is not shown. How to do that?
        h.quiver(res["m0"], res["mesh"], prefix+"m0.png", "1d m0")
        h.quiver(res["exc"], res["mesh"], prefix+"finmag.png", "1d finmag")
        h.quiver(res["oommf_exc"], res["oommf_mesh"], prefix+"oommf.png", "1d oommf")
        h.quiver(res["rel_diff"], res["oommf_mesh"], prefix+"rel_diff.png", "1d rel diff")
        h.boxplot(res["rel_diff"], prefix+"rel_diff_box.png")
