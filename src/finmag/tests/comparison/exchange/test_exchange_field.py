import os
import dolfin as df
import numpy as np
from dolfin import Interval
from finmag.sim.llg import LLG
from finmag.sim.helpers import quiver, vectors, norm, stats

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

x0 = 0; x1 = 20e-9; xn = 10;
Ms = 0.86e6; A = 1.3e-11

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0)
    mx = (2 * x - 1) * 2/3 
    mz = np.sin(2 * np.pi * x) / 2
    my = np.sqrt(1.0 - mx**2 - mz**2)
    return np.array([mx, my, mz])

class TestExchange():
    def setup_class(self, plot=False):
        mesh = Interval(xn, x0, x1)
        llg = LLG(mesh)

        coords = np.array(zip(* mesh.coordinates()))
        m0 = m_gen(coords).flatten()
        if plot:
            quiver(m0, mesh, MODULE_DIR + "m0_finmag.png")

        llg.set_m(m0)
        llg.Ms = Ms
        llg.A = A
        llg.setup(use_exchange=True)

        self.mesh = mesh
        self.V = llg.V
        self.m = llg.m
        H_exc = df.Function(llg.V)
        H_exc.vector()[:] = llg.exchange.compute_field()
        if plot:
            quiver(H_exc.vector().array(), mesh, MODULE_DIR + "exc_finmag.png")
        self.H_exc = H_exc

    def test_nmag(self):
        REL_TOLERANCE = 2e-14

        m_ref = np.genfromtxt(MODULE_DIR + "m0_nmag.txt")
        m_computed = vectors(self.m)
        assert m_ref.shape == m_computed.shape

        H_ref = np.genfromtxt(MODULE_DIR + "H_exc_nmag.txt")
        H_computed = vectors(self.H_exc.vector().array())
        assert H_ref.shape == H_computed.shape

        assert m_ref.shape == H_ref.shape
        m_cross_H_ref = np.cross(m_ref, H_ref)
        m_cross_H_computed = np.cross(m_computed, H_computed)

        diff = np.abs(m_cross_H_ref - m_cross_H_computed)
        rel_diff = diff/max([norm(v) for v in m_cross_H_ref])
      
        print "comparison with nmag, m x H, relative difference:"
        print stats(rel_diff)
        assert np.max(rel_diff) < REL_TOLERANCE

    def test_oommf(self):
        REL_TOLERANCE = 5e-8

        from finmag.util.oommf import mesh, oommf_uniform_exchange
        from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf

        oommf_mesh = mesh.Mesh((xn, 1, 1), size=(x1, 1e-12, 1e-12))
        oommf_exc  = oommf_uniform_exchange(oommf_m0(m_gen, oommf_mesh), Ms, A).flat
        finmag_exc = finmag_to_oommf(self.H_exc, oommf_mesh, dims=1)

        assert oommf_exc.shape == finmag_exc.shape
        diff = np.abs(oommf_exc - finmag_exc)
        rel_diff = diff / np.max(oommf_exc[0]**2 + oommf_exc[1]**2 + oommf_exc[2]**2)

        print "comparison with oommf, H, relative_difference:"
        print stats(rel_diff)
        assert np.max(rel_diff) < REL_TOLERANCE

    def test_magpar(self):
        pass

if __name__ == '__main__':
    t = TestExchange()
    t.setup_class(plot=True)
    t.test_nmag()
    t.test_oommf()
    t.test_magpar()
