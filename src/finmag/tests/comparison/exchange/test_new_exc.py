import os
import dolfin as df
import numpy as np
from finmag.energies.exchange import Exchange
from finmag.sim.helpers import vectors, norm, stats

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

x0 = 0; x1 = 20e-9; xn = 10;
Ms = 0.86e6; A = 1.3e-11

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0)
    mx = (2 * x - 1) * 2/3 
    mz = np.sin(2 * np.pi * x) / 2
    my = np.sqrt(1.0 - mx**2 - mz**2)
    return np.array([mx, my, mz])

def setup_finmag():
    mesh = df.Interval(xn, x0, x1)
    coords = np.array(zip(* mesh.coordinates()))

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    m = df.Function(S3)
    m.vector()[:] = m_gen(coords).flatten()

    exchange = Exchange(A)  
    exchange.setup(S3, m, Ms)

    H_exc = df.Function(S3)
    H_exc.vector()[:] = exchange.compute_field()
    return dict(m=m.vector().array(), H=H_exc)

def pytest_funcarg__finmag(request):
    finmag = request.cached_setup(setup=setup_finmag, scope="module")
    return finmag

def test_against_nmag(finmag):
    REL_TOLERANCE = 2e-14

    m_ref = np.genfromtxt(MODULE_DIR + "m0_nmag.txt")
    m_computed = vectors(finmag["m"])
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(MODULE_DIR + "H_exc_nmag.txt")
    H_computed = vectors(finmag["H"].vector().array())
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    diff = np.abs(m_cross_H_ref - m_cross_H_computed)
    rel_diff = diff/max([norm(v) for v in m_cross_H_ref])
 
    print "comparison with nmag, m x H, relative difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    f = setup_finmag()
    test_against_nmag(f)
