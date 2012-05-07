import os
import dolfin as df
import numpy as np
from finmag.util.convert_mesh import convert_mesh
from finmag.energies.anisotropy import UniaxialAnisotropy
from finmag.sim.helpers import vectors, stats, sphinx_sci as s

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

Ms = 0.86e6; K1 = 520e3; a = (1, 0, 0);
x1 = y1 = z1 = 20; # same as in bar_5_5_5.geo file

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0) # x, y and z as a fraction
    y = np.maximum(np.minimum(r[1]/y1, 1.0), 0.0) # between 0 and 1 
    z = np.maximum(np.minimum(r[2]/z1, 1.0), 0.0)
    mx = (2 - y) * (2 * x - 1) / 4
    mz = (2 - y) * (2 * z - 1) / 4 
    my = np.sqrt(1 - mx**2 - mz**2)
    return np.array([mx, my, mz])

def setup_finmag():
    mesh = df.Mesh(convert_mesh(MODULE_DIR + "/bar_5_5_5.geo"))
    coords = np.array(zip(* mesh.coordinates()))

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    m = df.Function(S3)
    m.vector()[:] = m_gen(coords).flatten()

    anisotropy = UniaxialAnisotropy(K1, a, Ms) 
    anisotropy.setup(S3, m, unit_length=1e-9)

    H_anis = df.Function(S3)
    H_anis.vector()[:] = anisotropy.compute_field()
    return dict(m=m.vector().array(), H=H_anis)

def pytest_funcarg__finmag(request):
    finmag = request.cached_setup(setup=setup_finmag, scope="module")
    return finmag

def test_nmag(finmag):
    REL_TOLERANCE = 6e-2

    m_ref = np.genfromtxt(MODULE_DIR + "m0_nmag.txt")
    m_computed = vectors(finmag["m"])
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(MODULE_DIR + "H_anis_nmag.txt")
    H_computed = vectors(finmag["H"].vector().array())
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    print "finmag m x H:"
    print m_cross_H_computed
    print "nmag m x H:"
    print m_cross_H_ref

    mxH = m_cross_H_computed.flatten()
    mxH_ref = m_cross_H_ref.flatten()
    diff = np.abs(mxH - mxH_ref)
    print "comparison with nmag, m x H, difference:"
    print stats(diff)

    rel_diff = diff/ np.sqrt(np.max(mxH_ref[0]**2 + mxH_ref[1]**2 + mxH_ref[2]**2))

    print "comparison with nmag, m x H, relative difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    f = setup_finmag()
    test_nmag(f)
