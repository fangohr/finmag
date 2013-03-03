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

params = df.Parameters("fk_demag")
poisson_params = df.Parameters("poisson")
poisson_params.add("method", "default")
poisson_params.add("preconditioner", "default")
laplace_params = df.Parameters("laplace")
laplace_params.add("method", "default")
laplace_params.add("preconditioner", "default")
params.add(poisson_params)
params.add(laplace_params)

# TODO: Add unit tests.
# TODO: Add benchmark to document once we get faster than existing implementation.


class FKDemag(object):
    # TODO: Add documentation.
    # TODO: Add timings provided by finmag.util.timings.
    def __init__(self):
        # TODO: Add way to change solver parameters.
        # Either here or after the instance was created. Maybe both.
        pass

    def setup(self, S3, m, Ms, unit_length=1):
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
        self._nodal_volumes = df.assemble(self._test1 * df.dx).array()
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
        self._bem, self._b2g_map = compute_bem_fk(OrientedBoundaryMesh(mesh))
        self._phi_1 = df.Function(self.S1)  # solution of inhomogeneous Neumann problem
        self._phi_2 = df.Function(self.S1)  # solution of Laplace equation inside domain
        self._phi = df.Function(self.S1)  # magnetic potential phi_1 + phi_2
        self._D = df.assemble(self.Ms * df.inner(self._trial3, df.grad(self._test1)) * df.dx)
        self._setup_field()

    def compute_field(self):
        """
        Compute the demagnetising field.

        *Returns*
            numpy.ndarray
                The demagnetising field.

        """
        self._compute_magnetic_potential()
        return self._compute_field()

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

    def energy_density_function(self):
        """
        Returns the energy density in the field as a dolfin function to allow probing.

        *Returns*
            dolfin.Function
                The energy density of the demagnetising field.

        """
        self._nodal_E_func.vector()[:] = self.energy_density()
        return self._nodal_E_func

    def _poisson_matrix(self):
        A = df.dot(df.grad(self._trial1), df.grad(self._test1)) * df.dx
        return df.assemble(A)  # stiffness matrix for Poisson equation

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

    def _setup_field(self):
        # TODO: This is the magpar method. Document how it works.
        a = df.inner(df.grad(self._trial1), self._test3) * df.dx
        b = df.dot(self._test3, df.Constant((-1, -1, -1))) * df.dx
        self.G = df.assemble(a)
        self.L = df.assemble(b).array()

    def _compute_field(self):
        # TODO: Write down how we would achieve the same result using df.project, albeit more slowly.
        H = self.G * self._phi.vector()
        return H.array() / self.L

if __name__ == "__main__":
    from finmag.util.meshes import sphere

    mesh = sphere(r=1.0, maxh=0.1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.Function(S3)
    m.assign(df.Constant((1, 0, 0)))
    Ms = 1

    demag = FKDemag()
    demag.setup(S3, m, Ms, unit_length=1e-9)

    H = demag.compute_field().reshape((3, -1))
    H_expected = np.array([-1.0 / 3.0, 0, 0])
    print "Computation of demagnetising field of uniformly magnetised sphere with Fredkin-Koehler method."
    print "Result H_d =\n\t{},\nshould be close to\n\t{}.".format(H.mean(1), H_expected)
    diff = np.max(np.abs(H.mean(1) - H_expected))
    print "Maximum difference to analytical result: {:.1e}.".format(diff)
    assert diff < 1e-3
