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
from finmag.util.consts import mu0
from finmag.native.llg import compute_bem_fk
from finmag.util.meshes import nodal_volume
from finmag.util.timings import Timings, default_timer, timed, mtimed

fk_timer = Timings()


class FKDemag(object):
    """
    Computation of the demagnetising field using the Fredkin-Koehler hybrid FEM/BEM technique.

    Fredkin, D.R. and Koehler, T.R., "`Hybrid method for computing demagnetizing fields`_",
    IEEE Transactions on Magnetics, vol.26, no.2, pp.415-417, Mar 1990.

    .. _Hybrid method for computing demagnetizing fields: http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=106342

    """
    def __init__(self):
        """
        Create a new FKDemag instance.

        The attribute `parameters` is a dict that contains the settings for the solvers
        for the Neumann (potential phi_1) and Laplace (potential phi_2) problems.

        Setting the method used by the solvers:
        Change the entries `phi_1_solver` and `phi_2_solver` to a value from
        `df.list_krylov_solver_methods()`. Default is dolfin's default.

        Setting the preconditioners:
        Change the entries `phi_1_preconditioner` and `phi_2_preconditioner` to
        a value from `df.list_krylov_solver_preconditioners()`. Default is dolfin's default.

        Setting the tolerances:
        Change the existing entries inside `phi_1` and `phi_2` which are themselves dicts.
        You can add new entries to these dicts as well. Everything which is
        understood by `df.KrylovSolver` is valid.

        """
        default_parameters = {
            'absolute_tolerance': 1e-6,
            'relative_tolerance': 1e-6,
            'maximum_iterations': int(1e4)
        }
        self.parameters = {
                'phi_1_solver': 'default',
                'phi_1_preconditioner': 'default',
                'phi_1': default_parameters,
                'phi_2_solver': 'default',
                'phi_2_preconditioner': 'default',
                'phi_2': default_parameters.copy()
        }

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
        self.m = m
        self.Ms = Ms
        self.unit_length = unit_length

        mesh = S3.mesh()
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.S3 = S3
        self.dim = mesh.topology().dim()

        self._test1 = df.TestFunction(self.S1)
        self._trial1 = df.TrialFunction(self.S1)
        self._test3 = df.TestFunction(self.S3)
        self._trial3 = df.TrialFunction(self.S3)

        # for computation of energy
        self._nodal_volumes = nodal_volume(self.S1, unit_length)
        self._H_func = df.Function(S3)  # we will copy field into this when we need the energy
        self._E_integrand = -0.5 * mu0 * df.dot(self._H_func, self.m * self.Ms)
        self._E = self._E_integrand * df.dx
        self._nodal_E = df.dot(self._E_integrand, self._test1) * df.dx
        self._nodal_E_func = df.Function(self.S1)

        # for computation of field and scalar magnetic potential
        self._poisson_matrix = self._poisson_matrix()
        self._poisson_solver = df.KrylovSolver(self._poisson_matrix,
                self.parameters['phi_1_solver'], self.parameters['phi_1_preconditioner'])
        self._poisson_solver.parameters.update(self.parameters['phi_1'])
        self._laplace_zeros = df.Function(self.S1).vector()
        self._laplace_solver = df.KrylovSolver(
                self.parameters['phi_2_solver'], self.parameters['phi_2_preconditioner'])
        self._laplace_solver.parameters.update(self.parameters['phi_2'])
        self._laplace_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True
        with timed('compute BEM', self.__class__.__name__, fk_timer):
            self._bem, self._b2g_map = compute_bem_fk(df.BoundaryMesh(mesh, 'exterior', False))
        self._phi_1 = df.Function(self.S1)  # solution of inhomogeneous Neumann problem
        self._phi_2 = df.Function(self.S1)  # solution of Laplace equation inside domain
        self._phi = df.Function(self.S1)  # magnetic potential phi_1 + phi_2

        # To be applied to the vector field m as first step of computation of _phi_1.
        # This gives us div(M), which is equal to Laplace(_phi_1), equation
        # which is then solved using _poisson_solver.
        self._Ms_times_divergence = df.assemble(self.Ms * df.inner(self._trial3, df.grad(self._test1)) * df.dx)

        self._setup_gradient_computation()

    @mtimed(default_timer)
    def compute_potential(self):
        """
        Compute the magnetic potential.

        *Returns*
            df.Function
                The magnetic potential.

        """
        self._compute_magnetic_potential()
        return self._phi

    @mtimed(default_timer)
    def compute_field(self):
        """
        Compute the demagnetising field.

        *Returns*
            numpy.ndarray
                The demagnetising field.

        """
        self._compute_magnetic_potential()
        return self._compute_gradient()

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
        g_1 = self._Ms_times_divergence * self.m.vector()
        self._poisson_solver.solve(self._phi_1.vector(), g_1)

        # compute _phi_2 on the boundary using the Dirichlet boundary
        # conditions we get from BEM * _phi_1 on the boundary.
        phi_1 = self._phi_1.vector()[self._b2g_map]
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
    def _setup_gradient_computation(self):
        """
        Prepare the discretised gradient to use in :py:meth:`FKDemag._compute_gradient`.

        We don't need the gradient field as a continuous field, we are only
        interested in the values at specific points. It is thus a waste of
        computational effort to use a projection of the gradient field, since
        it performs the fairly large operation of assembling a matrix and
        solving a linear system of equations.

        """
        A = df.inner(self._test3, - df.grad(self._trial1)) * df.dx
        # This can be applied to scalar functions.
        self._gradient = df.assemble(A)

        # The `A` above is in fact not quite the gradient, since we integrated
        # over the volume as well. We will divide by the volume later, after
        # the multiplication of the scalar magnetic potential. Since the two
        # operations are symmetric (multiplying by volume, dividing by volume)
        # we don't have to care for the units, i.e. unit_length.
        b = df.dot(self._test3, df.Constant((1, 1, 1))) * df.dx
        self._nodal_volumes_S3_no_units = df.assemble(b).array()

    @mtimed(fk_timer)
    def _compute_gradient(self):
        """
        Get the demagnetising field from the magnetic scalar potential.

        .. math::

            \\vec{H}_{\\mathrm{d}} = - \\nabla \\phi (\\vec{r})

        Using dolfin, we would translate this to

        .. sourcecode::

            H_d = df.project(- df.grad(self._phi), self.S3)

        but the method used here is computationally less expensive.

        """
        H = self._gradient * self._phi.vector()
        return H.array() / self._nodal_volumes_S3_no_units
