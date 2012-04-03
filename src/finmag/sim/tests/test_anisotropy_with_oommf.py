import dolfin as df
import numpy as np
from finmag.util.oommf import mesh
from finmag.util.oommf.comparison import compare_anisotropy
from finmag.sim.helpers import stats

K1 = 45e4 # J/m^3

REL_TOLERANCE = 5e-3 

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

    def m_gen(rs):
      rn = len(rs[0])
      return np.array([np.ones(rn), np.zeros(rn), np.zeros(rn)])
  
    from finmag.util.oommf.comparison import compare_anisotropy
    return compare_anisotropy(m_gen, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, name="single")

def one_dimensional_problem():
    x_min = 0; x_max = 10e-9; x_n = 200;
    dolfin_mesh = df.Interval(x_n, x_min, x_max)
    oommf_mesh = mesh.Mesh((20, 1, 1), size=(x_max, 1e-12, 1e-12))

    def m_gen(rs):
      xs = rs[0]
      return np.array([2*xs/x_max - 1, np.sqrt(1 - (2*xs/x_max - 1)**2), np.zeros(len(xs))])

    return compare_anisotropy(m_gen, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, dims=1, name="1D")

def three_dimensional_problem():
    x_max = 20e-9; y_max = 10e-9; z_max = 10e-9;
    dolfin_mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 100, 10, 10)
    oommf_mesh = mesh.Mesh((20, 10, 10), size=(x_max, y_max, z_max))

    def m_gen(rs):
      xs = rs[0]
      return np.array([np.sin(1e9 * xs/3)**2, np.zeros(len(xs)), np.ones(len(xs))])

    from finmag.util.oommf.comparison import compare_anisotropy
    return compare_anisotropy(m_gen, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, dims=3, name="3D")

if __name__ == '__main__':
    res0 = small_problem()
    print "0D problem, relative difference:\n", stats(res0["rel_diff"])
    res1 = one_dimensional_problem()
    print "1D problem, relative difference:\n", stats(res1["rel_diff"])
    res3 = three_dimensional_problem()
    print "3D problem, relative difference:\n", stats(res3["rel_diff"])
