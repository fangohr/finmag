"""Solvers for the demagnetization field using the Garcia-Cervera-Roma approach"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import logging
import dolfin as df
import solver_base as sb
import numpy as np
from finmag.util.timings import timings
from finmag.native.llg import compute_bem_gcr, OrientedBoundaryMesh
logger = logging.getLogger(name='finmag')

import finmag.util.solver_benchmark as bench

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

    This vector is assembled in the method build_vector_q using the box method

    .. math::
        q(\\vec r_i) \\approx  \\frac{ \\int_{supp(\\psi_i )} n \\cdotp
        (\\nabla \\phi_a - M ) \\psi_i dx}
        {\\int_{supp(\\psi_i)} \\psi_i dx} \\qquad \\qquad (6)

        
    Where :math:`\psi_i` is the basis function associated with :math:`i`.
    Currently the assembly is done over the entire mesh, even though only the values
    on the boundary are needed. Optimization is welcome here.
    
    The values of the boundary element matrix :math:`\\mathbf{B}` is given by

    .. math::

        B_{ij} = \\frac{1}{4\\pi}\\int_{\\Omega_j} \\psi_j(\\vec r)\\frac{1}
        {\\lvert \\vec R_i - \\vec r \\rvert} \\mathrm{d}s. \\qquad \\qquad (7)

    Solving the Laplace equation inside the domain and adding the two
    potentials, is also done in the exact same way as for the Fredkin-Koehler
    approach.

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
            set to True to run a benchmark of linear solvers
    """

    def __init__(self, mesh,m, parameters=None, degree=1, element="CG",
                 project_method='magpar', unit_length=1, Ms = 1.0,bench = False):
        
        #Initialize the base class
        #New interface have mesh,m,Ms
        sb.FemBemDeMagSolver.__init__(self,mesh,m,degree = degree, element=element,
                                             project_method = project_method,
                                             unit_length = unit_length,Ms = Ms,bench = bench)
        self.__name__ = "GCR Demag Solver"
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
        timings.startnext("Build boundary element matrix")
        self.boundary_mesh = OrientedBoundaryMesh(self.mesh)
        self.bem, self.b2g = compute_bem_gcr(self.boundary_mesh)
        timings.stop("Build boundary element matrix")

        #Buffer Surface Node Areas for the box method
        self.surface_node_areas = df.assemble(self.v*df.ds, mesh=self.mesh).array()+1e-300

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
        if self.bench:
            bench.solve(self.poisson_matrix_dirichlet,phia.vector(),F, benchmark = True)
           # df.solve(self.poisson_matrix_dirichlet,phia.vector(),F,"gmres","ilu")
        else:
            timings.startnext("1st linear solve")
            self.poisson_solver.solve(phia.vector(),F)
            timings.stop("1st linear solve")
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
        
        q = q_dot_v/self.surface_node_areas
        return q

#Gabriel TODO add the exact q calculation back to the class.
class ExactQBuilder():
    def get_dof_normal_dict_avg(self,normtionary):
        """
        Provides a dictionary with all of the boundary DOF's as keys
        and an average of facet normal components associated to the DOF as values
        V = FunctionSpace
        """
        #Take an average of the normals in normtionary
        avgnormtionary = {k:np.array([ float(sum(i))/float(len(i)) for i in zip(*normtionary[k])]) for k in normtionary}
        #Renormalize the normals
        avgnormtionary = {k: avgnormtionary[k]/df.sqrt(np.dot(avgnormtionary[k],avgnormtionary[k].conj())) for k in avgnormtionary}
        return avgnormtionary

    def build_boundary_data(self):
        """
        Builds two boundary data dictionaries
        1.doftionary key- dofnumber, value - coordinates
        2.normtionary key - dofnumber, value - average of all facet normal components associated to a DOF
        """
        mesh = self.V.mesh()
        #Initialize the mesh data
        mesh.init()
        d = mesh.topology().dim()
        dm = self.V.dofmap()
        boundarydofs = self.get_boundary_dofs(self.V)

        #It is very import that this vector has the right length
        #It holds the local dof numbers associated to a facet
        facetdofs = np.zeros(dm.num_facet_dofs(),dtype=np.uintc)

        #Initialize dof-to-normal dictionary
        doftonormal = {}
        doftionary = {}
        #Loop over boundary facets
        for facet in df.facets(mesh):
            cells = facet.entities(d)
            #one cell means we are on the boundary
            if len(cells) ==1:
                #######################################
                #Shared Data for Normal and coordinates
                #######################################

                #create one cell (since we have CG)
                cell = df.Cell(mesh,cells[0])
                #Local to global map
                globaldofcell = dm.cell_dofs(cells[0])

                #######################################
                #Find  Dof Coordinates
                #######################################

                #Create the cell dofs and see if any
                #of the global numbers turn up in BoundaryDofs
                #If so update doftionary with the coordinates
                celldofcord = dm.tabulate_coordinates(cell)

                for locind,dof in enumerate(globaldofcell):
                    if dof in boundarydofs:
                        doftionary[dof] = celldofcord[locind]

                #######################################
                #Find Normals
                #######################################
                local_fi = cell.index(facet)
                dm.tabulate_facet_dofs(facetdofs,local_fi)
                #Global numbers of facet dofs
                globaldoffacet = [globaldofcell[ld] for ld in facetdofs]
                #add the facet's normal to every dof it contains
                for gdof in globaldoffacet:
                    n = facet.normal()
                    ntup = tuple([n[i] for i in range(d)])
                    #If gdof not in dictionary initialize a list
                    if gdof not in doftonormal:
                        doftonormal[gdof] = []
                    #Prevent redundancy in Normals (for example 3d UnitCube CG1)
                    if ntup not in doftonormal[gdof]:
                        doftonormal[gdof].append(ntup)

            elif len(cells) == 2:
                #we are on the inside so continue
                continue
            else:
                assert 1==2,"Expected only two cells per facet and not " + str(len(cells))

        #Build the average normtionary and save data
        self.doftonormal = doftonormal
        self.normtionary = self.get_dof_normal_dict_avg(doftonormal)
        self.doftionary = doftionary
        #numpy array with type double for use by instant (c++)
        self.doflist_double = np.array(doftionary.keys(),dtype = self.normtionary[self.normtionary.keys()[0]].dtype.name)
        self.bdofs = np.array(doftionary.keys())

if __name__ == "__main__":
    from finmag.demag.problems import prob_fembem_testcases as pft
    from finmag.sim import helpers
    
    problem = pft.MagSphereBase(2.0, 10)
    kwargs = problem.kwargs()
##    kwargs["bench"] = True

    #Make a more interesting m
    m = df.interpolate(df.Expression(["x[0]*x[1]+3", "x[2]+5", "x[1]+7"]),
                       df.VectorFunctionSpace(problem.mesh,"CG",1))
    
    m.vector()[:] = helpers.fnormalise(m.vector().array())
    kwargs["m"] = m
    solver = FemBemGCRSolver(**kwargs)
    sol = solver.solve()
    print timings
    df.plot(sol)
    df.plot(solver.phia, title = "phia")
    df.interactive()
