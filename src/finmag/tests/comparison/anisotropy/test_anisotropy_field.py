import os
import dolfin as df
import numpy as np
from finmag.util.convert_mesh import convert_mesh
from finmag.sim.llg import LLG
from finmag.sim.helpers import quiver, vectors, norm, stats

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

Ms = 0.86e6; K1 = 520e3; a = (1, 0, 0);

x1 = y1 = z1 = 5; # same as in bar_5_5_5.geo file

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0) # x, y and z as a fraction
    y = np.maximum(np.minimum(r[1]/y1, 1.0), 0.0) # between 0 and 1 in the 
    z = np.maximum(np.minimum(r[2]/z1, 1.0), 0.0)
    mx = (2 - y) * (2 * x - 1) / 4
    mz = (2 - y) * (2 * z - 1) / 4 
    my = np.sqrt(1 - mx**2 - mz**2)
    return np.array([mx, my, mz])

class TestAnisotropy():
    def setup_class(self, plot=False):
        mesh = df.Mesh(convert_mesh(MODULE_DIR + "/bar_5_5_5.geo"))
        llg = LLG(mesh, mesh_units=1e-9)
        coords = np.array(zip(* mesh.coordinates()))
        m0 = m_gen(coords).flatten()
        if plot:
            quiver(m0, mesh, MODULE_DIR + "m0_finmag.png")

        llg.set_m(m0)
        llg.Ms = Ms
        llg.add_uniaxial_anisotropy(K1, df.Constant(a))
        llg.setup(use_exchange=False)

        self.mesh = mesh
        self.V = llg.V
        self.m = llg.m
        self._m = llg._m
        H_anis = df.Function(llg.V)
        H_anis.vector()[:] = llg._anisotropies[0].compute_field()
        if plot:
            quiver(H_anis.vector().array(), mesh, MODULE_DIR + "anis_finmag.png")
        self.H_anis = H_anis

    def test_nmag(self):
        REL_TOLERANCE = 0.4 # ???

        m_ref = np.genfromtxt(MODULE_DIR + "m0_nmag.txt")
        m_computed = vectors(self.m)
        assert m_ref.shape == m_computed.shape

        H_ref = np.genfromtxt(MODULE_DIR + "H_anis_nmag.txt")
        H_computed = vectors(self.H_anis.vector().array())
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
        from finmag.util.oommf import mesh, oommf_uniaxial_anisotropy
        from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf

        REL_TOLERANCE = 2e-5

        oommf_mesh = mesh.Mesh((2, 2, 2), size=(5e-9, 5e-9, 5e-9))
        oommf_anis  = oommf_uniaxial_anisotropy(oommf_m0(lambda r: m_gen(np.array(r)*1e9), oommf_mesh), Ms, K1, a).flat
        finmag_anis = finmag_to_oommf(self.H_anis, oommf_mesh, dims=3)

        assert oommf_anis.shape == finmag_anis.shape
        diff = np.abs(oommf_anis - finmag_anis)
        rel_diff = diff / np.max(oommf_anis[0]**2 + oommf_anis[1]**2 + oommf_anis[2]**2)

        print "comparison with oommf, H, relative_difference:"
        print stats(rel_diff)
        assert np.max(rel_diff) < REL_TOLERANCE

    def test_magpar(self):
        from finmag.tests.magpar.magpar import compute_anis_magpar, compare_field_directly

        REL_TOLERANCE = 8e-6

        magpar_nodes, magpar_anis = compute_anis_magpar(self.V, self._m, K1, a, Ms, mesh_units=1)
        _, _, diff, rel_diff = compare_field_directly(
                self.mesh.coordinates(), self.H_anis.vector().array(),
                magpar_nodes, magpar_anis)
        print "comparison with magpar, H, relative_difference:"
        print stats(rel_diff)
        assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    t = TestAnisotropy()
    t.setup_class(plot=False)
    t.test_nmag()
    t.test_oommf()
    t.test_magpar()
