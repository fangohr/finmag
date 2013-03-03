"""
Computation of the demagnetising field using the Fredkin-Koehler
technique and the infamous magpar method.

Rationale: The existing implementation in FemBemFKSolver (child class
of FemBemDeMagSolver) is kind of a mess. This does the same thing in the same
time with less code. Should be more conducive to further optimisation or as
a template for other techniques like the GCR.

"""
import numpy as np
import dolfin as df
import finmag.util.timings as timings
from finmag.util.consts import mu0
from finmag.native.llg import OrientedBoundaryMesh, compute_bem_fk
from finmag.util.meshes import nodal_volume
from finmag.util.timings import Timings, default_timer, timed, mtimed

params = df.Parameters("fk_demag")
poisson_params = df.Parameters("poisson")
poisson_params.add("method", "default")
poisson_params.add("preconditioner", "default")
laplace_params = df.Parameters("laplace")
laplace_params.add("method", "default")
laplace_params.add("preconditioner", "default")
params.add(poisson_params)
params.add(laplace_params)

fk_timer = Timings()

# TODO: Add benchmark to document once we get faster than existing implementation.


class FKDemag(object):
    # TODO: Add documentation.
    def __init__(self):
        """
        Create a new FKDemag instance.

        """
        # TODO: Add way to change solver parameters.
        # Either here or after the instance was created. Maybe both.
        pass

    @mtimed(default_timer)
    def setup(self, S3, m, Ms, unit_length=1):
        """
        Setup the FKDemag instance. Usually called automatically by the Simulation object.

        *Arguments*

        S3: dolfin.VectorFunctionSpace

            The finite element space the magnetisation is defined on.

        m: dolfin.Function on S3

            The unit magnetisation.

        Ms: float

            The saturation magnetisation in A/m.

        unit_length: float

            The length (in m) represented by one unit on the mesh. Default 1.

        """
        # TODO: Find more meaningful names for some attributes like _D.
        # TODO: Overthink liberal use of _ prefix for attribute names.
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length

        # related to mesh
        mesh = S3.mesh()
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = S3
        self.dim = mesh.topology().dim()

        # test and trial functions
        self._test1 = df.TestFunction(self.S1)
        self._trial1 = df.TrialFunction(self.S1)
        self._test3 = df.TestFunction(self.S3)
        self._trial3 = df.TrialFunction(self.S3)

        # for computation of energy
        self._nodal_volumes = nodal_volume(self.S1, unit_length)
        self._H_func = df.Function(S3)  # we will copy field into this when we need the energy
        self._E_integrand = -0.5 * mu0 * df.dot(self._H_func, self.Ms * self.m)
        self._E = self._E_integrand * df.dx
        self._nodal_E = df.dot(self._E_integrand, self._test1) * df.dx
        self._nodal_E_func = df.Function(self.S1)

        # for computation of field
        self._poisson_matrix = self._poisson_matrix()
        self._poisson_solver = df.KrylovSolver(self._poisson_matrix, params["poisson"]["method"], params["poisson"]["preconditioner"])
        self._laplace_zeros = df.Function(self.S1).vector()
        self._laplace_solver = df.KrylovSolver(params["laplace"]["method"], params["laplace"]["preconditioner"])
        self._laplace_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True
        with timed('compute BEM', self.__class__.__name__, fk_timer):
            self._bem, self._b2g_map = compute_bem_fk(OrientedBoundaryMesh(mesh))
        self._phi_1 = df.Function(self.S1)  # solution of inhomogeneous Neumann problem
        self._phi_2 = df.Function(self.S1)  # solution of Laplace equation inside domain
        self._phi = df.Function(self.S1)  # magnetic potential phi_1 + phi_2
        self._D = df.assemble(self.Ms * df.inner(self._trial3, df.grad(self._test1)) * df.dx)
        self._setup_field()

    @mtimed(default_timer)
    def compute_field(self):
        """
        Compute the demagnetising field.

        *Returns*
            numpy.ndarray
                The demagnetising field.

        """
        self._compute_magnetic_potential()
        return self._compute_field()

    @mtimed(default_timer)
    def compute_energy(self):
        """
        Compute the total energy of the field.

        .. math::

            E_\\mathrm{d} = -\\frac12 \\mu_0 \\int_\\Omega
            H_\\mathrm{d} \\cdot \\vec M \\mathrm{d}x

        *Returns*
            Float
                The energy of the demagnetising field.

        """
        self._H_func.vector()[:] = self.compute_field()
        return df.assemble(self._E) * self.unit_length ** self.dim

    @mtimed(default_timer)
    def energy_density(self):
        """
        Compute the energy density in the field.

        .. math::
            \\rho = \\frac{E_{\\mathrm{d}, i}}{V_i},

        where V_i is the volume associated with the node i.

        *Returns*
            numpy.ndarray
                The energy density of the demagnetising field.

        """
        self._H_func.vector()[:] = self.compute_field()
        nodal_E = df.assemble(self._nodal_E).array() * self.unit_length ** self.dim
        return nodal_E / self._nodal_volumes

    @mtimed(default_timer)
    def energy_density_function(self):
        """
        Returns the energy density in the field as a dolfin function to allow probing.

        *Returns*
            dolfin.Function
                The energy density of the demagnetising field.

        """
        self._nodal_E_func.vector()[:] = self.energy_density()
        return self._nodal_E_func

    @mtimed(fk_timer)
    def _poisson_matrix(self):
        A = df.dot(df.grad(self._trial1), df.grad(self._test1)) * df.dx
        return df.assemble(A)  # stiffness matrix for Poisson equation

    @mtimed(fk_timer)
    def _compute_magnetic_potential(self):
        # compute _phi_1 on the whole domain
        g_1 = self._D * self.m.vector()
        self._poisson_solver.solve(self._phi_1.vector(), g_1)

        # _phi_1 restricted to the boundary
        phi_1 = self._phi_1.vector()[self._b2g_map]

        # compute _phi_2 on the boundary
        self._phi_2.vector()[self._b2g_map[:]] = np.dot(self._bem, phi_1.array())

        # compute _phi_2 inside the domain
        boundary_condition = df.DirichletBC(self.S1, self._phi_2, df.DomainBoundary())
        A = self._poisson_matrix.copy()
        b = self._laplace_zeros
        boundary_condition.apply(A, b)
        self._laplace_solver.solve(A, self._phi_2.vector(), b)

        # add _phi_1 and _phi_2 to obtain magnetic potential
        self._phi.vector()[:] = self._phi_1.vector() + self._phi_2.vector()

    @mtimed(fk_timer)
    def _setup_field(self):
        # TODO: This is the magpar method. Document how it works.
        a = df.inner(df.grad(self._trial1), self._test3) * df.dx
        b = df.dot(self._test3, df.Constant((-1, -1, -1))) * df.dx
        self.G = df.assemble(a)
        self.L = df.assemble(b).array()

    @mtimed(fk_timer)
    def _compute_field(self):
        # TODO: Write down how we would achieve the same result using df.project, albeit more slowly.
        H = self.G * self._phi.vector()
        return H.array() / self.L
