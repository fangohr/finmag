import numpy as np
import dolfin as df
from finmag.sim.helpers import stats 
from finmag.util.oommf import mesh
from finmag.util.oommf.comparison import compare_exchange 

def test_one_dimensional_problem():
    REL_TOLERANCE = 2
    results = one_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_three_dimensional_problem():
    REL_TOLERANCE = 7
    results = three_dimensional_problem()
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def one_dimensional_problem():
    x_min = 0; x_max = 100e-9; x_n = 50;

    dolfin_mesh = df.Interval(x_n, x_min, x_max)
    oommf_mesh = mesh.Mesh((x_n, 1, 1), size=(x_max, 1e-12, 1e-12))

    def m_gen(coords):
        xs = coords[0]
        return np.array([np.sqrt(xs/x_max), np.sqrt(1 - xs/x_max), np.zeros(len(xs))])

    return compare_exchange(m_gen, dolfin_mesh, oommf_mesh, dims=1, name="1d")

def three_dimensional_problem():
    x_max = 100e-9; y_max = z_max = 10e-9;
    x_n = 50; y_n = z_n = x_n/10;

    dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, x_n, x_n, x_n)
    oommf_mesh = mesh.Mesh((x_n, y_n, z_n), size=(x_max, y_max, z_max))

    def m_gen(coords):
        xs = coords[0]
        return np.array([np.sqrt(xs/x_max), np.sqrt(1 - xs/x_max), np.zeros(len(xs))])

    return compare_exchange(m_gen, dolfin_mesh, oommf_mesh, name="3d")

if __name__ == '__main__':
    res1 = one_dimensional_problem()
    print "1D problem, relative difference:\n", stats(res1["rel_diff"])
    res3 = three_dimensional_problem()
    print "3D problem, relative difference:\n", stats(res3["rel_diff"])
