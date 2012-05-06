import unittest
import numpy as np
import math
import dolfin as df
from finmag.native.llg import compute_bem_element, compute_bem, OrientedBoundaryMesh
from finmag.util import time_counter

from finmag.demag import belement_magpar
from finmag.sim.llg import LLG

compute_belement = belement_magpar.return_bele_magpar()

def compute_belement_magpar(r1, r2, r3):
    res = np.zeros(3)
    compute_belement(np.zeros(3), np.array(r1, dtype=float), np.array(r2, dtype=float), np.array(r3, dtype=float), res)
    return res

class BemComputationTests(unittest.TestCase):
    def test_simple(self):
        r1 = np.array([1., 0., 0.])
        r2 = np.array([2., 1., 3.])
        r3 = np.array([5., 0., 1.])
        be_magpar = compute_belement_magpar(r1, r2, r3)
        be_native = compute_bem_element(r1, r2, r3)
        print "Magpar: ", be_magpar
        print "Native C++: ", be_native
        self.assertAlmostEqual(np.max(np.abs(be_magpar - be_native)), 0, delta=1e-12)

    def test_cell_ordering(self):
        mesh = df.UnitCube(1,1,1)
        centre = np.array([0.5, 0.5, 0.5])
        boundary_mesh = df.BoundaryMesh(mesh)
        coordinates = boundary_mesh.coordinates()
        for i in xrange(boundary_mesh.num_cells()):
            cell = df.Cell(boundary_mesh, i)
            p1 = coordinates[cell.entities(0)[0]]
            p2 = coordinates[cell.entities(0)[1]]
            p3 = coordinates[cell.entities(0)[2]]
            n = np.cross(p2 - p1, p3 - p1)
            print "Boundary face %d, normal orientation %g" % (i, np.sign(np.dot(n, p1-centre)))

    def run_bem_computation_test(self, mesh):
        llg = LLG(mesh)
        llg.set_m((1, 0, 0))
        llg.Ms = 1.
        llg.setup(use_demag=True)
        bem_finmag = llg.demag.bem
        bem_native = np.zeros(bem_finmag.shape)
        bem, b2g = compute_bem(OrientedBoundaryMesh(mesh))
        g2finmag = llg.demag.gnodes_to_bnodes
        for i_dolfin in xrange(bem.shape[0]):
            i_finmag = g2finmag[b2g[i_dolfin]]

            for j_dolfin in xrange(bem.shape[0]):
                j_finmag = g2finmag[b2g[j_dolfin]]
                bem_native[i_finmag, j_finmag] = bem[i_dolfin, j_dolfin]
        if np.max(np.abs(bem_finmag - bem_native)) > 1e-12:
            print "Finmag:", np.round(bem_finmag, 4)
            print "Native:", np.round(bem_native, 4)
            print "Difference:", np.round(bem_native - bem_finmag, 4)
            self.fail("Finmag and native computation of BEM differ, mesh: " + str(mesh))

    def test_bem_computation(self):
        self.run_bem_computation_test(df.UnitSphere(1))
        self.run_bem_computation_test(df.UnitSphere(2))
        self.run_bem_computation_test(df.UnitSphere(3))
        self.run_bem_computation_test(df.UnitSphere(4))
        self.run_bem_computation_test(df.UnitSphere(6))
        self.run_bem_computation_test(df.UnitCube(3,3,3))

    def test_bem_perf(self):
        mesh = df.UnitCube(15, 15, 15)
        boundary_mesh = OrientedBoundaryMesh(mesh)
        c = time_counter.counter()
        while c.next():
            OrientedBoundaryMesh(mesh)
        print "Boundary mesh computation for %s: %s" % (mesh, c)
        c = time_counter.counter()
        while c.next():
            bem, _ = compute_bem(boundary_mesh)
            n = bem.shape[0]
        print "BEM computation for %dx%d (%.2f Mnodes/sec): %s" % (n, n, c.calls_per_sec(n*n/1e6), c)

    def test_bem_netgen(self):
        netgen_mesh = df.Mesh("bem_netgen_test_mesh.xml.gz")
        bem, b2g_map = compute_bem(OrientedBoundaryMesh(netgen_mesh))


