import dolfin as df
import numpy as np
from aeon import Timer
from finmag.util import helpers
import finmag.util.solver_benchmark as bench

# Define default parameters for the fembem solvers
default_parameters = df.Parameters("demag_options")
poisson = df.Parameters("poisson_solver")
poisson.add("method", "default")
poisson.add("preconditioner", "default")
laplace = df.Parameters("laplace_solver")
laplace.add("method", "default")
laplace.add("preconditioner", "default")
default_parameters.add(poisson)
default_parameters.add(laplace)

demag_timings = Timer()


class FemBemDeMagSolver(object):
    """Base Class for FEM/BEM Demag Solvers containing shared methods
        for a top level demag solver interface see
        class Demag in finmag/energies/demag

        *Arguments*
        mesh
            dolfin Mesh object
        m
            the Dolfin object representing the (unit) magnetisation
        Ms
            the saturation magnetisation
        parameters
            dolfin.Parameters of method and preconditioner to linear solvers
            If not specified the defualt parameters contained in solver_base.py
            are used.
        degree
            polynomial degree of the function space
        element
            finite element type, default is "CG" or Lagrange polynomial.
        unit_length
            the scale of the mesh, defaults to 1.
        project_method
            method to calculate the demag field from the potential
            possible methods are
                * 'magpar'
                * 'project'
        bench
            set to True to run a benchmark of linear solvers

    """

    def __init__(self, mesh,m, parameters=None, degree=1, element="CG", project_method='magpar',
                 unit_length=1, Ms=1.0, bench=False, normalize=True, solver_type=None):
        #Problem objects and parameters
        self.name = "Demag"
        self.in_jacobian = False
        self.mesh = mesh
        self.unit_length = unit_length
        self.degree = degree
        self.bench = bench
        self.parameters = parameters

        #This is used in energy density calculations
        self.mu0 = np.pi*4e-7 # Vs/(Am)

        #Mesh Facet Normal
        self.n = df.FacetNormal(self.mesh)

        #Spaces and functions for the Demag Potential
        self.V = df.FunctionSpace(self.mesh,element,degree)
        self.v = df.TestFunction(self.V)
        self.u = df.TrialFunction(self.V)
        self.phi = df.Function(self.V)

        #Space and functions for the Demag Field
        self.W = df.VectorFunctionSpace(self.mesh, element, degree, dim=3)
        self.w = df.TrialFunction(self.W)
        self.vv = df.TestFunction(self.W)
        self.H_demag = df.Function(self.W)

        #Interpolate the Unit Magentisation field if necessary
        #A try block was not used since it might lead to an unneccessary (and potentially bad)
        #interpolation
        if isinstance(m, df.Expression) or isinstance(m, df.Constant):
            self.m = df.interpolate(m,self.W)

        elif isinstance(m,tuple):
            self.m = df.interpolate(df.Expression(m),self.W)

        elif isinstance(m,list):
            self.m = df.interpolate(df.Expression(tuple(m)),self.W)

        else:
            self.m = m
        #Normalize m (should be normalized anyway).
        if normalize:
            self.m.vector()[:] = helpers.fnormalise(self.m.vector().array())

        self.Ms = Ms

        # Initilize the boundary element matrix variable
        self.bem = None

        #Objects that are needed frequently for linear solves.
        self.poisson_matrix = self.build_poisson_matrix()
        self.laplace_zeros = df.Function(self.V).vector()

        #2nd FEM.
        if parameters:
            method = parameters["laplace_solver"]["method"]
            pc = parameters["laplace_solver"]["preconditioner"]
        else:
            method, pc = "default", "default"

        if solver_type is None:
            solver_type = 'Krylov'
        solver_type = solver_type.lower()
        if solver_type == 'lu':
            self.laplace_solver = df.LUSolver()
            self.laplace_solver.parameters["reuse_factorization"] = True
        elif solver_type == 'krylov':
            self.laplace_solver = df.KrylovSolver(method, pc)
            # We're setting 'same_nonzero_pattern=True' to enforce the
            # same matrix sparsity pattern across different demag solves,
            # which should speed up things.
            try:
                # Old syntax (dolfin version <= 1.2)
                self.laplace_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True
            except KeyError:
                # New syntax (dolfin version >= 1.2.0+)
                self.laplace_solver.parameters["preconditioner"]["structure"] = "same_nonzero_pattern"
                from finmag.util.helpers import warn_about_outdated_code
                warn_about_outdated_code(
                    min_dolfin_version='1.3.0',
                    msg='There is outdated code in FKDemag.__init__() setting the '
                        '"same_nonzero_pattern" parameter which might be good to '
                        'remove since the syntax has changed in recent dolfin versions.')
        else:
            raise ValueError("Wrong solver type specified: '{}' (allowed values: 'Krylov', 'LU')".format(solver_type))

        #Objects needed for energy density computation
        self.nodal_vol = df.assemble(self.v*df.dx, mesh=self.mesh).array()
        self.ED = df.Function(self.V)

        #Method to calculate the Demag field from the potential
        self.project_method = project_method
        if self.project_method == 'magpar':
            self.__setup_field_magpar()
            self.__compute_field = self.__compute_field_magpar
        elif self.project_method == 'project':
            self.__compute_field = self.__compute_field_project
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'magpar',
                                    * 'project'""")
    def solve():
        return

    def compute_field(self):
        """
        Compute the demag field.

        .. note::
            The interface has to be changed to this later anyway, so
            we can just keep it this way so we don't need to change the
            examples later.

        *Returns*
            numpy.ndarray
                The demag field.

        """
        self.solve()
        return self.__compute_field()

    def compute_potential(self):
        self.solve()
        return self.phi

    def scalar_potential(self):
        """Return the scalar potential."""
        return self.phi

    def compute_energy(self):
        """
        Compute the demag energy defined by

        .. math::

            E_\\mathrm{demag} = -\\frac12 \\mu_0 \\int_\\Omega
            H_\\mathrm{demag} \\cdot \\vec M \\mathrm{d}x

        *Returns*
            Float
                The demag energy.

        """
        self.H_demag.vector()[:] = self.compute_field()
        E = -0.5*self.mu0*df.dot(self.H_demag, self.m*self.Ms)*df.dx
        return df.assemble(E, mesh=self.mesh)*\
                self.unit_length**self.mesh.topology().dim()

    def energy_density(self):
        """
        Compute the demag energy density,

        .. math::

            \\frac{E_\\mathrm{demag}}{V},

        where V is the volume of each node.

        *Returns*
            numpy.ndarray
                The demag energy density.

        """
        self.H_demag.vector()[:] = self.compute_field()
        E = df.dot(-0.5*self.mu0*df.dot(self.H_demag, self.m*self.Ms), self.v)*df.dx
        nodal_E = df.assemble(E).array()
        return nodal_E/self.nodal_vol

    def energy_density_function(self):
        """
        Compute the demag energy density the same way as the
        function above, but return a Function to allow probing.

        *Returns*
            dolfin.Function
                The demag energy density.
        """
        self.ED.vector()[:] = self.energy_density()
        return self.ED

    def build_poisson_matrix(self):
        """assemble a poisson equation 'stiffness' matrix"""
        a = df.dot(df.grad(self.u),df.grad(self.v))*df.dx
        return df.assemble(a)

    def solve_laplace_inside(self, function, solverparams=None):
        """Take a functions boundary data as a dirichlet BC and solve
            a laplace equation"""
        bc = df.DirichletBC(self.V, function, df.DomainBoundary())
        A = self.poisson_matrix.copy()
        b = self.laplace_zeros#.copy()
        bc.apply(A, b)

        if self.bench:
            bench.solve(A,function.vector(),b,benchmark = True)
        else:
            demag_timings.start("2nd linear solve", self.__class__.__name__)
            self.laplace_iter = self.laplace_solver.solve(A, function.vector(), b)
            demag_timings.stop("2nd linear solve", self.__class__.__name__)
        return function

    def __compute_field_project(self):
        """
        Dolfin method of projecting the scalar potential
        onto a dolfin.VectorFunctionSpace.
        """
        Hdemag = df.project(-df.grad(self.phi), self.W)
        return Hdemag.vector().array()

    def __setup_field_magpar(self):
        """Needed by the magpar method we may use instead of project."""
        #FIXME: Someone with a bit more insight in this method should
        # write something about it in the documentation.
        a = df.inner(df.grad(self.u), self.vv)*df.dx
        b = df.dot(self.vv, df.Constant([-1, -1, -1]))*df.dx
        self.G = df.assemble(a)
        self.L = df.assemble(b).array()

    def __compute_field_magpar(self):
        """Magpar method used by Weiwei."""
        Hd = self.G*self.phi.vector()
        Hd = Hd.array()/self.L
        return Hd

    def get_demagfield(self,phi = None,use_default_function_space = True):
        """
        Returns the projection of the negative gradient of
        phi onto a DG0 space defined on the same mesh
        Note: Do not trust the viper solver to plot the DeMag field,
        it can give some wierd results, paraview is recommended instead

        use_default_function_space - If true project into self.W,
                                     if false project into a Vector DG0 space
                                     over the mesh of phi.
        """
        if phi is None:
            phi = self.phi

        Hdemag = -df.grad(phi)
        if use_default_function_space == True:
            Hdemag = df.project(Hdemag,self.W)
        else:
            if self.D == 1:
                Hspace = df.FunctionSpace(phi.function_space().mesh(),"DG",0)
            else:
                Hspace = df.VectorFunctionSpace(phi.function_space().mesh(),"DG",0)
            Hdemag = df.project(Hdemag,Hspace)
        return Hdemag
