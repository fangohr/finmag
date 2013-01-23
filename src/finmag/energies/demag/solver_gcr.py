import logging
import dolfin as df
import solver_base as sb
import numpy as np
from finmag.native.llg import compute_bem_gcr
logger = logging.getLogger(name='finmag')

import finmag.util.solver_benchmark as bench
from solver_gcr_qvector_pe import PEQBuilder

gcr_timings = sb.demag_timings

class FemBemGCRSolver(sb.FemBemDeMagSolver,PEQBuilder):
    """
    This approach is similar to the :py:class:FKSolver <finmag.demag.solver_fk.FemBemFKSolver>`
    approach, so we will just comment on the differences between the
    approaches. As before, the magnetic scalar potential is diveded into
    two parts, :math:`\\phi = \\phi_a + \\phi_b`, but the definition of these
    are different. :math:`\\phi_a` is the solution of the inhomogeneous
    Dirichlet problem defined as

    .. math::

        \\Delta \\phi_a(\\vec r) = \\nabla \\cdotp \\vec M(\\vec r) \\qquad \\qquad (1)

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

    This vector can be assembled in two ways, using the default point evaluation method,
    or the box method.

    The formula for the point evaluation method is almost the same as the definition, except that
    :math 'n' has been replaced by :math: '\\tilde(n) = \\frac{\\ sum n_i}{\\| sum n_i \\|}',
    the normalized average of the norms associated to all of the facets that meet at
    a vertex on the boundary.
    
    .. math::
        q(\\vec r) = -\\tilde(n) \\cdot \\vec M(\\vec r) +
        \\frac{\\partial \\phi_a}{\\partial n} \\qquad \\qquad (5)

    This method has the advantage of offering better precision (add a link to nmag 1 example),
    but is slower at the moment since it is implemented in python.

    Q vector assembly with the box method is done by calling dolfin assemble() using the formula
    for the definition of q multiplied by a test function. This gives an estimate of q(i) that is averaged
    by the integral of a basis function, whose volume is divided out afterwards. 

    .. math::
        q(\\vec r_i) \\approx  \\frac{ \\int_{supp(\\psi_i )} n \\cdotp
        (\\nabla \\phi_a - M ) \\psi_i dx}
        {\\int_{supp(\\psi_i)} \\psi_i dx} \\qquad \\qquad (6)

        
    Where :math:`\psi_i` is the basis function associated with :math:`i`.
    Currently the assembly is done over the entire mesh, even though only the values    on the boundary are needed. Optimization is welcome here.
    
    The values of the boundary element matrix :math:`\\mathbf{B}` is given by

    .. math::

        B_{ij} = \\frac{1}{4\\pi}\\int_{\\Omega_j} \\psi_j(\\vec r)\\frac{1}
        {\\lvert \\vec R_i - \\vec r \\rvert} \\mathrm{d}s. \\qquad \\qquad (7)

    Solving the Laplace equation inside the domain and adding the two
    potentials, is also done in the exact same way as for the Fredkin-Koehler
    approach.

    Linear solver tolerances can be set by accessing the attributes
    laplace_solver or poisson_solver.

    .. code-block:: python

        demag = FemBemGCRSolver(mesh,m)
        demag.poisson_solver["method"] = "cg"
        demag.poisson_solver["preconditioner"] = "ilu"
        demag.laplace_solver["method"] = "cg"
        demag.laplace_solver["preconditioner"] = "ilu"

    A benchmark of all possible Krylov solver and preconditioner combinations
    can be run as follows.
    
    .. code-block:: python

        demag = FemBemGCRSolver(mesh,m,benchmark = True)
        demag.solve()

    after a solve the number of krylov iterations can be accessed via the attributes
    laplace_iter, poisson_iter
    
    .. code-block:: python

        demag = FemBemGCRSolver(mesh,m)
        demag.solve()
        print demag.laplace_iter
        print demag.poisson_iter
        


    *For an interface more inline with the rest of FinMag Code please use
    the wrapper class Demag in finmag/energies/demag.*


    *Arguments*
        mesh
            dolfin Mesh object
        m
            the Dolfin object representing the (unit) magnetisation
        Ms
            the saturation magnetisation 
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
        bench
            set to True to run a benchmark of linear solvers.
        qvector method
            method to assemble the vector q. Choices are "pe" for point evaluation
            or "box" for the box method.
    """
    def __init__(self, mesh,m, parameters=sb.default_parameters, degree=1, element="CG",
                 project_method='magpar', unit_length=1, Ms = 1.0,bench = False,
                 qvector_method = 'pe'):
        
        #Initialize the base class
        sb.FemBemDeMagSolver.__init__(self,mesh,m,parameters = parameters,degree = degree, element=element,
                                             project_method = project_method,
                                             unit_length = unit_length,Ms = Ms,bench = bench)
        self.__name__ = "GCR Demag Solver"
        self.qvector_method = qvector_method
        
        #Define the potentials
        self.phia = df.Function(self.V)
        self.phib = df.Function(self.V)

        #Buffer a homogeneous Dirichlet Poisson Matrix as well as BC
        self.phia_bc = df.DirichletBC(self.V,0,lambda x, on_boundary: on_boundary)
        self.poisson_matrix_dirichlet = self.poisson_matrix.copy() 
        self.phia_bc.apply(self.poisson_matrix_dirichlet)

        #Linear Solver parameters for the 1st solve
        if parameters:
            method = parameters["poisson_solver"]["method"]
            pc = parameters["poisson_solver"]["preconditioner"]
        else:
            method = "default"
            pc = "default"
        self.poisson_solver = df.KrylovSolver(self.poisson_matrix_dirichlet, method, pc)
        
        #Buffer the BEM
        gcr_timings.start_next(self.__class__.__name__, "build BEM")
        self.boundary_mesh = df.BoundaryMesh(self.mesh, False)
        self.bem, self.b2g = compute_bem_gcr(self.boundary_mesh)
        gcr_timings.stop(self.__class__.__name__, "build BEM")

        if self.qvector_method == "box":
            #Buffer Surface Node Areas for the box method
            self.surface_node_areas = df.assemble(self.v*df.ds, mesh=self.mesh).array()+1e-300
        elif self.qvector_method == "pe":
            #Build boundary data for the point evaluation q method
            self.build_boundary_data()
        else:
            raise Exception("Only 'box' and 'pe' are possible qvector_method values")
        
    def solve(self):
        """
        Solve for the Demag field using GCR and FemBem
        Potential is returned, demag field stored
        """

        #Solve for phia using FEM
        logger.debug("GCR: Solving for phi_a")
        gcr_timings.start_next(self.__class__.__name__, "Solve phia")
        self.phia = self.solve_phia(self.phia)
        
        #Assemble the vector q.
        logger.debug("GCR: Solving for phi_b on the boundary")
        gcr_timings.start_next(self.__class__.__name__, "Build q vector")
        if self.qvector_method == "pe":
            q = self.build_vector_q_pe(self.m,self.Ms,self.phia)
        elif self.qvector_method == "box":
            q = self.build_vector_q(self.m,self.Ms,self.phia)
        else:
            raise Exception("Only 'box' and 'pe' are possible qvector_method values")
        
        # Compute phi2 on boundary using the BEM matrix
        gcr_timings.start_next(self.__class__.__name__, "Compute phiab on the boundary")
        phib_boundary = np.dot(self.bem, q[self.b2g])

        #Insert the boundary data into the function phib.
        gcr_timings.start_next(self.__class__.__name__, "Inserting bem data into a dolfin function")
        self.phib.vector()[self.b2g] = phib_boundary
        
        #Solve for phib on the inside of the mesh with Fem, eq. (3)
        logger.debug("GCR: Solve for phi_b (laplace on the inside)")
        gcr_timings.start_next(self.__class__.__name__, "Compute phi_b on the inside")
        self.phib = self.solve_laplace_inside(self.phib)
        
        #Add together the two potentials
        gcr_timings.start_next(self.__class__.__name__, "Add phi1 and phi2")
        self.phi.vector()[:] = self.phia.vector() + self.phib.vector()
        gcr_timings.stop(self.__class__.__name__, "Add phi1 and phi2")

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
        if self.bench:
            bench.solve(self.poisson_matrix_dirichlet,phia.vector(),F, benchmark = True)
        else:
            gcr_timings.start_next(self.__class__.__name__, "1st linear solve")
            self.poisson_iter = self.poisson_solver.solve(phia.vector(),F)
            gcr_timings.stop(self.__class__.__name__, "1st linear solve")
        return phia
    
    def build_vector_q(self,m,Ms,phi1):
        """Get the left hand side q in the BEM equation phib = bem*qf
           using the box method. Assembly is done over the entire mesh,
           afterwards the global to boundary mapping is used to extract
           the relevant data"""
        
        q_dot_v = df.assemble(Ms*df.dot(self.n, -m + df.grad(phi1))*self.v*df.ds,
                              mesh=self.mesh).array()
    
        q = q_dot_v/self.surface_node_areas
        return q

if __name__ == "__main__":
    from finmag.tests.demag.problems import prob_fembem_testcases as pft
    from finmag.sim import helpers
    
    problem = pft.MagSphereBase(5.0, 10)

    #Make a more interesting m
    m = df.interpolate(df.Expression(["x[0]*x[1]+3", "x[2]+5", "x[1]+7"]),
                       df.VectorFunctionSpace(problem.mesh,"CG",1))
    
    m.vector()[:] = helpers.fnormalise(m.vector().array())
    m = df.Expression(("1.0","0.0","0.0"))
    solver = FemBemGCRSolver(problem.mesh,m,bench = False,qvector_method = "pe")
    sol = solver.solve()
    print gcr_timings
    df.plot(sol)
    df.plot(solver.phia, title = "phia")
    df.interactive()
