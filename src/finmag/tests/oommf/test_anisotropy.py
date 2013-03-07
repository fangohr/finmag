import dolfin as df
import numpy as np
from finmag.util.oommf import mesh
from finmag.util.oommf.comparison import compare_anisotropy
from finmag.util.helpers import stats

K1 = 45e4 # J/m^31
Ms = 0.86e6

def test_small_problem():
    results = small_problem()
    REL_TOLERANCE = 1e-15
    print "0d: rel_diff_max:",np.nanmax(results["rel_diff"])
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_one_dimensional_problem():
    results = one_dimensional_problem()
    REL_TOLERANCE = 1e-9 #for 100,000 FE nodes, 6e-4 for 200 nodes

    print "1d: rel_diff_max:",np.nanmax(results["rel_diff"])
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def test_three_dimensional_problem():
    results = three_dimensional_problem()
    REL_TOLERANCE = 9e-2
    print "3d: rel_diff_max:",np.nanmax(results["rel_diff"])
    assert np.nanmax(results["rel_diff"]) < REL_TOLERANCE

def small_problem():
    # The oommf mesh corresponding to this problem only has a single cell.
    x_max = 1e-9; y_max = 1e-9; z_max = 1e-9;
    dolfin_mesh = df.BoxMesh(0, 0, 0, x_max, y_max, z_max, 5, 5, 5)
    oommf_mesh = mesh.Mesh((1, 1, 1), size=(x_max, y_max, z_max))

    def m_gen(rs):
      rn = len(rs[0])
      return np.array([np.ones(rn), np.zeros(rn), np.zeros(rn)])
  
    from finmag.util.oommf.comparison import compare_anisotropy
    return compare_anisotropy(m_gen, Ms, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, name="single")

def one_dimensional_problem():
    x_min = 0; x_max = 100e-9; x_n = 100000;
    dolfin_mesh = df.IntervalMesh(x_n, x_min, x_max)
    oommf_mesh = mesh.Mesh((20, 1, 1), size=(x_max, 1e-12, 1e-12))

    def m_gen(rs):
      xs = rs[0]
      return np.array([xs/x_max, np.sqrt(1 - (xs/x_max)**2), np.zeros(len(xs))])

    return compare_anisotropy(m_gen, Ms, K1, (1, 1, 1), dolfin_mesh, oommf_mesh, dims=1, name="1D")

def three_dimensional_problem():
    x_max = 100e-9; y_max = z_max = 1e-9;
    x_n=20; y_n = z_n = 1;

    dolfin_mesh = df.BoxMesh(0, 0, 0, x_max, y_max, z_max, x_n, y_n, z_n)
    print dolfin_mesh.num_vertices()
    oommf_mesh = mesh.Mesh((x_n, y_n, z_n), size=(x_max, y_max, z_max))

    def m_gen(rs):
      xs = rs[0]
      return np.array([xs/x_max, np.sqrt(1 - (0.9*xs/x_max)**2 - 0.01), 0.1*np.ones(len(xs))])

    from finmag.util.oommf.comparison import compare_anisotropy
    return compare_anisotropy(m_gen, Ms, K1, (0, 0, 1), dolfin_mesh, oommf_mesh, dims=3, name="3D")

if __name__ == '__main__':
    res0 = small_problem()
    print "0D problem, relative difference:\n", stats(res0["rel_diff"])
    res1 = one_dimensional_problem()
    print "1D problem, relative difference:\n", stats(res1["rel_diff"])
    res3 = three_dimensional_problem()
    print "3D problem, relative difference:\n", stats(res3["rel_diff"])
