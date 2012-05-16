"""Solvers for the demagnetization field using the Garcia-Cervera-Roma approach"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import math
import logging
import dolfin as df
import finmag.demag.solver_base as sb
import numpy as np
from finmag.util.timings import timings
from finmag.native.llg import compute_bem_gcr, OrientedBoundaryMesh
logger = logging.getLogger(name='finmag')

class FemBemGCRSolver(sb.FemBemDeMagSolver):
    """
    This approach is similar to the :py:class:FKSolver <finmag.demag.solver_fk.FemBemFKSolver>`
    approach, so we will just comment on the differences between the
    approaches. As before, the magnetic scalar potential is diveded into
    two parts, :math:`\\phi = \\phi_a + \\phi_b`, but the definition of these
    are different. :math:`\\phi_a` is the solution of the inhomogeneous
    Dirichlet problem defined as

    .. math::

        \\Delta \\phi_a(\\vec r) = \\nabla \\vec M(\\vec r) \\qquad \\qquad (1)

    inside the domain, and

    .. math::

        \\phi_a(\\vec r) = 0 \\qquad \\qquad (2)

    on the boundary and outside the domain. This is solved in a similar
    manner as before, with the variational forms and boundary condition
    given by (code-block 1)

    .. code-block:: python

        a = dot(grad(u),grad(v))*dx
        f = (-div(self.M)*v)*dx  # Source term

        #Define Boundary Conditions
        bc = DirichletBC(V,0,"on_boundary")

    The second potential, :math:`\phi_b`, is the solution of the Laplace equation

    .. math::

        \Delta \phi_b = 0 \\quad \\qquad (3)

    inside the domain, its normal derivative has a discontinuity of

    .. math::

        \\bigtriangleup \\left(\\frac{\\partial \\phi_b}
        {\\partial n}\\right) = -n \\cdot \\vec M(\\vec r)
        + \\frac{\\partial \\phi_a}{\\partial n}

    on the boundary and it vanishes at infinity, with
    :math:`\\phi_b(\\vec r) \\rightarrow 0` for
    :math:`\\lvert \\vec r \\rvert \\rightarrow \\infty`.
    As for the Fredkin-Koehler approach, the boundary problem can be solved
    with BEM. Unlike the Fredkin-Koehler approach, where the vector equation
    for the second part of the potential is the product between the boundary
    element matrix and the first potential on the boundary, we now have

    .. math::

        \\Phi_b = \\mathbf{B} \\cdot \\vec Q \\qquad \\qquad (4)

    The vector :math:`\\vec Q` contains the potential values :math:`q` at
    the sites of the surface mesh, with :math:`q` defined as the right hand
    side of the boundary equation,

    .. math::

        q(\\vec r) = -n \\cdot \\vec M(\\vec r) +
        \\frac{\\partial \\phi_a}{\\partial n} \\qquad \\qquad (5)

    This vector is assembled in the function assemble_qvector_exact.
    The values of the boundary element matrix :math:`\\mathbf{B}` is given by

    .. math::

        B_{ij} = \\frac{1}{4\\pi}\\int_{\\Omega_j} \\psi_j(\\vec r)\\frac{1}
        {\\lvert \\vec R_i - \\vec r \\rvert} \\mathrm{d}s. \\qquad \\qquad (6)

    Solving the Laplace equation inside the domain and adding the two
    potentials, is also done in the exact same way as for the Fredkin-Koehler
    approach.

    *Arguments*
        problem
            An object of type DemagProblem
        degree
            polynomial degree of the function space
        element
            finite element type, default is "CG" or Lagrange polynomial.
        unit_length
            the scale of the mesh, defaults to 1.
        project_method
            possible methods are
                * 'magpar'
                * 'project'
        phiaTOL = df.e-12,
            relative tolerance of the first krylov linear solver
        phibTOL = df.e-12
            relative tolerance of the second krylov linear solver
    """

    def __init__(self, problem, degree=1, element="CG", project_method='magpar', unit_length=1,
                 phiaTOL = df.e-12,phibTOL = df.e-12):
        
        #Initialize the base class
        sb.FemBemDeMagSolver.__init__(self,problem,degree, element=element,
                                             project_method = project_method,
                                             unit_length = unit_length,phi2TOL = phibTOL)
        self.phiaTOL = phiaTOL

        #Define the potentials
        self.phia = df.Function(self.V)
        self.phib = df.Function(self.V)

        #Buffer a homogeneous Dirichlet Poisson Matrix as well as BC
        self.phia_bc = df.DirichletBC(self.V,0,lambda x, on_boundary: on_boundary)
        self.poisson_matrix_dirichlet = self.poisson_matrix.copy() 
        self.phia_bc.apply(self.poisson_matrix_dirichlet)

        #Linear Solver parameters for the 1st solve
        #Let dolfin decide what's best
        self.phia_solver = df.KrylovSolver(self.poisson_matrix_dirichlet)
        self.phia_solver.parameters["relative_tolerance"] = self.phiaTOL
        
        #Buffer the BEM
        timings.startnext("Build boundary element matrix")
        self.boundary_mesh = OrientedBoundaryMesh(self.mesh)
        self.bem, self.b2g = compute_bem_gcr(self.boundary_mesh)
        timings.stop("Build boundary element matrix")

    def solve(self):
        """
        Solve for the Demag field using GCR and FemBem
        Potential is returned, demag field stored
        """

        #Solve for phia using FEM
        logger.info("GCR: Solving for phi_a")
        timings.startnext("Solve phia")
        self.phia = self.solve_phia(self.phia)
        
        #Assemble the vector q.
        logger.info("GCR: Solving for phi_b on the boundary")
        timings.startnext("Build q vector")
        q = self.build_vector_q(self.m,self.Ms,self.phia)

        # Compute phi2 on boundary using the BEM matrix
        timings.startnext("Compute phiab on the boundary")
        phib_boundary = np.dot(self.bem, q[self.b2g])

        #Insert the boundary data into the function phib.
        self.phib.vector()[self.b2g] = phib_boundary
        
        #Solve for phib on the inside of the mesh with Fem, eq. (3)
        logger.info("GCR: Solve for phi_b (laplace on the inside)")
        timings.startnext("Compute phi_b on the inside")
        self.phib = self.solve_laplace_inside(self.phib)
        
        #Add together the two potentials
        timings.startnext("Add phi1 and phi2")
        self.phi.vector()[:] = self.phia.vector() + self.phib.vector()
        timings.stop("Add phi1 and phi2")

        return self.phi

    def solve_phia(self,phia):
        """
        Solve for potential phia in the Magentic region using FEM.
        """
        V = phia.function_space()

        #Source term depends on m (code-block 1 - second line)
        #So it should be recalculated at every time step.
        f = -self.Ms*(df.div(self.m)*self.v)*df.dx  #Source term
        F = df.assemble(f)
        self.phia_bc.apply(F)

        #Solve for phia
        self.phia_solver.solve(phia.vector(),F)
        #Replace with LU solve
        #df.solve(self.poisson_matrix_dirichlet,phia.vector(),F)
        
        return phia
    
    def build_vector_q(self,m,Ms,phi1):
        """Get the left hand side q in the BEM equation phib = bem*qf
           using the box method. Assembly is done over the entire mesh,
           afterwards the global to boundary mapping is used to extract
           the relevant data"""
        
        q_dot_v = df.assemble(Ms*df.dot(self.n, -m + df.grad(phi1))*self.v*df.ds,
                              mesh=self.mesh).array()
        
        surface_node_areas = df.assemble(self.v*df.ds, mesh=self.mesh).array()+1e-300
        q = q_dot_v/surface_node_areas
        return q

if __name__ == "__main__":
    from finmag.demag.problems import prob_fembem_testcases as pft
    problem = pft.MagSphere20()
    solver = FemBemGCRSolver(problem)
    sol = solver.solve()
    print timings
    df.plot(sol)
    df.interactive()
