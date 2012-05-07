import unittest
import numpy as np
import math
import dolfin as df
import os
from finmag.native.llg import compute_lindholm_L, compute_lindholm_K, compute_bem_fk, compute_bem_gcr, OrientedBoundaryMesh
from finmag.util import time_counter
from finmag.sim import helpers
from finmag.demag import belement_magpar
from finmag.sim.llg import LLG
from finmag.tests.solid_angle_invariance_tests import random_3d_rotation_matrix
from finmag.demag.problems.prob_fembem_testcases import MagSphere

compute_belement = belement_magpar.return_bele_magpar()

def compute_belement_magpar(r1, r2, r3):
    res = np.zeros(3)
    compute_belement(np.zeros(3), np.array(r1, dtype=float), np.array(r2, dtype=float), np.array(r3, dtype=float), res)
    return res

def compute_demag_solver():
    g1 = df.assemble(self.Ms * df.dot(self.n, self.m) * self.v * df.ds\
    - self.Ms * df.div(self.m) * self.v * df.dx)
    self.phi1_solver.solve(self.phi1.vector(), g1)

def normalise_phi(phi, mesh):
    volume = df.assemble(df.Constant(1) * df.dx, mesh=mesh)
    average = df.assemble(phi * df.dx, mesh=mesh)
    phi.vector()[:] = phi.vector().array() - average / volume

def compute_scalar_potential_llg(mesh, m_expr=df.Constant([1, 0, 0]), Ms=1.):
    llg = LLG(mesh)
    llg.set_m(m_expr)
    llg.Ms = Ms
    llg.setup(use_demag=True)
    llg.compute_H_eff()
    normalise_phi(llg.demag.phi, mesh)
    return llg.demag.phi

# Computes the derivative f'(x) using the finite difference approximation
def differentiate_fd(f, x, eps=1e-4, offsets=(-2,-1,1,2), weights=(1./12.,-2./3.,2./3.,-1./12.)):
    return sum(f(x+eps*dx)*weights[i] for i, dx in enumerate(offsets))/eps

PHI1_SOLVER_PARAMS = PHI2_SOLVER_PARAMS = {
    "linear_solver": "gmres",
    "preconditioner": "ilu",
}

# Solves the demag problem for phi using the FK method and the native BEM matrix
def compute_scalar_potential_native_fk(mesh, m_expr=df.Constant([1, 0, 0]), Ms=1.):
    # Set up the FE problems
    V_m = df.VectorFunctionSpace(mesh, 'Lagrange', 1, dim=3)
    V_phi = df.FunctionSpace(mesh, 'Lagrange', 1)
    u = df.TrialFunction(V_phi)
    v = df.TestFunction(V_phi)
    n = df.FacetNormal(mesh)
    m = df.interpolate(m_expr, V_m)
    m.vector()[:] = helpers.fnormalise(m.vector().array())

    phi1 = df.Function(V_phi)
    phi2_bc = df.Function(V_phi)
    phi2 = df.Function(V_phi)

    # Solve the variational problem for phi1
    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    L = -Ms * df.div(m) * v * df.dx + Ms * df.dot(n, m) * v * df.ds
    # The problem for phi1 is underconstrained since the solution is defined up to a constant.
    # However, the Krylov subspace methods are supposed to be regularising, therefore
    # as long as we use a fixed initial guess, or renormalise the initial guess
    # periodically during time integration, we should be fine
    df.solve(a == L, phi1, solver_parameters=PHI1_SOLVER_PARAMS)
    # Compute the BEM
    boundary_mesh = OrientedBoundaryMesh(mesh)
    bem, b2g = compute_bem_fk(boundary_mesh)
    # Restrict phi1 to boundary
    phi1_boundary = phi1.vector().array()[b2g]
    # Compute phi2 on boundary using the BEM matrix
    phi2_boundary = np.dot(bem, phi1_boundary)
    # Map phi2 back to global space
    phi2_bc.vector()[b2g] = phi2_boundary
    # Solve the laplace equation for phi2
    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    bc = df.DirichletBC(V_phi, phi2_bc, lambda x, on_boundary: on_boundary)
    L = df.Constant(0) * v * df.dx
    df.solve(a == L, phi2, bc, solver_parameters=PHI2_SOLVER_PARAMS)
    # Add phi2 back to phi1
    phi1.vector()[:] += phi2.vector().array()
    normalise_phi(phi1, mesh)
    return phi1

# Solves the demag problem for phi using the GCR method and the native BEM matrix
def compute_scalar_potential_native_gcr(mesh, m_expr=df.Constant([1, 0, 0]), Ms=1.):
    # Set up the FE problems
    V_m = df.VectorFunctionSpace(mesh, 'Lagrange', 1, dim=3)
    V_phi = df.FunctionSpace(mesh, 'Lagrange', 1)
    u = df.TrialFunction(V_phi)
    v = df.TestFunction(V_phi)
    n = df.FacetNormal(mesh)
    fa = df.FacetArea(mesh)
    m = df.interpolate(m_expr, V_m)
    m.vector()[:] = helpers.fnormalise(m.vector().array())

    phi1 = df.Function(V_phi)
    phi2_bc = df.Function(V_phi)
    phi2 = df.Function(V_phi)

    # Solve the variational problem for phi1
    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    L = -Ms * df.div(m) * v * df.dx
    phi1_bc = df.DirichletBC(V_phi, df.Constant(0), lambda x, on_boundary: on_boundary)
    df.solve(a == L, phi1, phi1_bc, solver_parameters=PHI1_SOLVER_PARAMS)
    # Compute the BEM
    boundary_mesh = OrientedBoundaryMesh(mesh)
    bem, b2g = compute_bem_gcr(boundary_mesh)
    # Compute the surface monopole density with something like the box method
    # This is might be wildly inaccurate for geometries with corners,
    # but what else can we do apart from making BEM depend both on M
    # and phi thus using 4x as much RAM
    # P.S. looks like this is inaccurate since this does not work...
    # How can I test this?
    q = df.assemble(Ms*df.dot(n, -m + df.grad(phi1))*v/fa*df.ds, mesh=mesh).array()
    # Compute phi2 on boundary using the BEM matrix
    phi2_boundary = np.dot(bem, q[b2g])
    # Map phi2 back to global space
    phi2_bc.vector()[b2g] = phi2_boundary
    # Solve the laplace equation for phi2
    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    bc = df.DirichletBC(V_phi, phi2_bc, lambda x, on_boundary: on_boundary)
    L = df.Constant(0) * v * df.dx
    df.solve(a == L, phi2, bc, solver_parameters=PHI2_SOLVER_PARAMS)
    # Add phi2 back to phi1
    phi1.vector()[:] += phi2.vector().array()
    normalise_phi(phi1, mesh)
    return phi1

class BemComputationTests(unittest.TestCase):
    def test_simple(self):
        r1 = np.array([1., 0., 0.])
        r2 = np.array([2., 1., 3.])
        r3 = np.array([5., 0., 1.])
        be_magpar = compute_belement_magpar(r1, r2, r3)
        be_native = compute_lindholm_L(np.zeros(3), r1, r2, r3)
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
        bem, b2g = compute_bem_fk(OrientedBoundaryMesh(mesh))
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
            bem, _ = compute_bem_fk(boundary_mesh)
            n = bem.shape[0]
        print "FK BEM computation for %dx%d (%.2f Mnodes/sec): %s" % (n, n, c.calls_per_sec(n*n/1e6), c)
        c = time_counter.counter()
        while c.next():
            bem, _ = compute_bem_gcr(boundary_mesh)
        print "GCR BEM computation for %dx%d (%.2f Mnodes/sec): %s" % (n, n, c.calls_per_sec(n*n/1e6), c)

    def test_bem_netgen(self):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        netgen_mesh = df.Mesh(os.path.join(module_dir, "bem_netgen_test_mesh.xml.gz"))
        bem, b2g_map = compute_bem_fk(OrientedBoundaryMesh(netgen_mesh))

    def run_demag_computation_test(self, mesh, m_expr, compute_func, method_name, tol=1e-10, ref=compute_scalar_potential_llg):
        phi_a = ref(mesh, m_expr)
        phi_b = compute_func(mesh, m_expr)
        error = df.errornorm(phi_a, phi_b, mesh=mesh)
        message = "Method: %s, mesh: %s, m: %s, error: %8g" % (method_name, mesh, m_expr, error)
        print message
#        print phi_a.vector().array()
#        print phi_b.vector().array()
        self.assertAlmostEqual(error, 0, delta=tol, msg="Error is above threshold %g, %s" % (tol, message))

    def test_compute_scalar_potential_fk(self):
        m1 = df.Constant([1, 0, 0])
        m2 = df.Expression(["x[0]*x[1]+3", "x[2]+5", "x[1]+7"])
        for k in xrange(1,5+1):
            self.run_demag_computation_test(df.UnitSphere(k), m1, compute_scalar_potential_native_fk, "native, FK")
            self.run_demag_computation_test(df.UnitSphere(k), m2, compute_scalar_potential_native_fk, "native, FK")
        self.run_demag_computation_test(MagSphere(1, 0.25).mesh, m1, compute_scalar_potential_native_fk, "native, FK")
        self.run_demag_computation_test(MagSphere(1, 0.25).mesh, m2, compute_scalar_potential_native_fk, "native, FK")

    def atest_compute_scalar_potential_gcr(self):
        m1 = df.Constant([1, 0, 0])
        m2 = df.Expression(["x[0]*x[1]+3", "x[2]+5", "x[1]+7"])
        self.run_demag_computation_test(MagSphere(1, 0.1).mesh, m1, compute_scalar_potential_native_gcr, "native, GCR", ref=compute_scalar_potential_native_fk)
        for k in xrange(5,10+1):
#            self.run_demag_computation_test(df.UnitCube(k,k,k), m1, compute_scalar_potential_native_gcr, "native, GCR", ref=compute_scalar_potential_native_fk)
            self.run_demag_computation_test(df.UnitSphere(k), m1, compute_scalar_potential_native_gcr, "native, GCR", ref=compute_scalar_potential_native_fk)
#            self.run_demag_computation_test(df.UnitSphere(k), m2, compute_scalar_potential_native_gcr, "native, GCR")

    def run_symmetry_test(self, formula):
        func = globals()[formula]
        np.random.seed(1)
        for i in xrange(100):
            r = np.random.rand(3)*2-1
            r1 = np.random.rand(3)*2-1
            r2 = np.random.rand(3)*2-1
            r3 = np.random.rand(3)*2-1
            a = np.random.rand(3)*2-1
            b = random_3d_rotation_matrix(1)[0]
            # Compute the formula
            v1 = func(r, r1, r2, r3)
            # The formula is invariant under rotations via b
            v2 = func(np.dot(b, r), np.dot(b, r1), np.dot(b, r2), np.dot(b, r3))
            self.assertAlmostEqual(np.max(np.abs(v1 - v2)), 0)
            # and under translations via a
            v3 = func(r+a, r1+a, r2+a, r3+a)
            self.assertAlmostEqual(np.max(np.abs(v1 - v3)), 0)
            # and under permutations of r1-r2-r3
            v4 = func(r, r2, r3, r1)
            self.assertAlmostEqual(np.max(np.abs(v1[[1,2,0]] - v4)), 0)

    def test_lindholm_L_symmetry(self):
        # Lindholm formulas should be invariant under rotations and translations
        self.run_symmetry_test("compute_lindholm_L")

    def test_lindholm_K_symmetry(self):
        self.run_symmetry_test("compute_lindholm_K")

    def test_lindholm_derivative(self):
        # the double layer potential is the derivative of the single layer potential
        # with respect to displacements of the triangle in the normal direction
        np.random.seed(1)
        for i in xrange(100):
            r = np.random.rand(3) * 2 - 1
            r1 = np.random.rand(3) * 2 - 1
            r2 = np.random.rand(3) * 2 - 1
            r3 = np.random.rand(3) * 2 - 1
            zeta  = np.cross(r2-r1, r3-r1)
            zeta /= np.linalg.norm(zeta)
            def f(x):
                return compute_lindholm_K(r, r1+x*zeta, r2+x*zeta, r3+x*zeta)
            L1 = compute_lindholm_L(r, r1, r2, r3)
            L2 = differentiate_fd(f, 0)
            self.assertAlmostEqual(np.max(np.abs(L1 - L2)), 0, delta=1e-10)


