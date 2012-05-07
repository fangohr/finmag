import logging
import numpy as np
import dolfin as df
from finmag.demag import solver_base as sb
from finmag.util.timings import timings
from finmag.util.progress_bar import ProgressBar
from finmag.native.llg import OrientedBoundaryMesh, compute_bem_fk

logger = logging.getLogger(name='finmag')

__all__ = ["FemBemFKSolver"]
class FemBemFKSolver(sb.FemBemDeMagSolver):
    """
    The idea of the Fredkin-Koehler approach is to split the magnetic
    potential into two parts, :math:`\\phi = \\phi_1 + \\phi_2`.

    :math:`\\phi_1` solves the inhomogeneous Neumann problem

    .. math::

        \\Delta \\phi_1 = \\nabla \\cdot \\vec M(\\vec r), \\quad \\vec r \\in \\Omega, \\qquad
        \\qquad

    with

    .. math::

        \\frac{\partial \\phi_1}{\\partial \\vec n} = \\vec n \\cdot \\vec M \\qquad \\qquad

    on :math:`\\Gamma`. In addition, :math:`\\phi_1(\\vec r) = 0` for
    :math:`\\vec r \\not \\in \\Omega`.

    Multiplying with a test function, :math:`v`, and integrate over the domain,
    the corresponding variational problem reads

    .. math::

        \\int_\\Omega \\nabla \\phi_1 \\cdot \\nabla v =
        \\int_{\\partial \\Omega} (\\vec n \\cdot vec M) v \\mathrm{ds} -
        \\int_\\Omega (\\nabla \\cdot \\vec
        M)v \\mathrm{d}x \\qquad \\qquad (1)

    .. note::

        Might be a -1 error here, but I think not.

    This could be solved straight forward by (code-block 1)

    .. code-block:: python

        a = df.inner(df.grad(u), df.grad(v))*df.dx
        L = self.Ms*df.div(m)*self.v*df.dx
        df.solve(a==L, self.phi1)

    but we are instead using the fact that L can be written as (code-block 2)

    .. code-block:: python

        b = Ms*df.inner(w, df.grad(v))*df.dx
        D = df.assemble(b)
        L = D*m.vector()

    In this way, we can assemble D at setup, and do not have to
    recompute it each time. This speeds up the solver significantly.

    :math:`\\phi_2` is the solution of Laplace's equation inside the domain,

    .. math::

        \\Delta \\phi_2(\\vec r) = 0
        \\quad \\hbox{for } \\vec r \\in \\Omega. \\qquad \\qquad (2)

    At the boundary, :math:`\\phi_2` has a discontinuity of

    .. math::

        \\bigtriangleup \\phi_2(\\vec r) = \\phi_1(\\vec r), \\qquad \\qquad

    and it disappears at infinity, i.e.

    .. math::

        \\phi_2(\\vec r) \\rightarrow 0 \\quad \\mathrm{for}
        \\quad \\lvert \\vec r \\rvert \\rightarrow \\infty. \\qquad \\qquad

    In contrast to the Poisson equation for :math:`\\phi_1`,
    which is solved straight forward in a finite domain, we now need to
    apply a BEM technique to solve the equations for :math:`\\phi_2`.
    First, we solve the equation on the boundary. By eq. (2.51) in Knittel's
    thesis, this yieds

    .. math::

        \\Phi_2 = \\mathbf{B} \\cdot \\Phi_1, \\qquad \\qquad (3)

    with :math:`\\Phi_1` as the vector of elements from :math:`\\phi_1` which
    is on the boundary. These are found by the the global-to-boundary mapping

    .. code-block:: python

        Phi1 = self.phi1.vector().array()[g2b_map]

    The elements of the boundary element matrix
    :math:`\\mathbf{B}` are given by

    .. math::

        B_{ij} = \\frac{1}{4\\pi}\\int_{\\Gamma_j} \\psi_j(\\vec r)
        \\frac{(\\vec R_i - \\vec r) \\cdot n(\\vec r)}
        {\\lvert \\vec R_i - \\vec r \\rvert^3} \\mathrm{d}s +
        \\left(\\frac{\\Omega(\\vec R_i)}{4\\pi} - 1 \\right) \\delta_{ij}. \\qquad \\qquad (4)

    Here, :math:`\\psi` is a set of basis functions and
    :math:`\\Omega(\\vec R)` denotes the solid angle.

    Having both :math:`\\Phi_1` and :math:`\\mathbf{B}`,
    we use numpy.dot to compute the dot product.

    .. code-block:: python

        self.Phi2 = np.dot(self.bem, Phi1)

    Now that we have obtained the values of :math:`\\phi_2` on the boundary,
    we need to solve the Laplace equation inside the domain, with
    these boundary values as boundary condition. This is done
    straight forward in Dolfin, as we can use the DirichletBC class.
    First we fill in the boundary values in the phi2 function at the
    right places.

    .. code-block:: python

        self.phi2.vector().array()[g2b_map] = self.Phi2

    And this can now be applied to DirichletBC to create boundary
    conditions. Remember that A is our previously assembled Poisson matrix,
    and b is here a zero vector. The complete code then reads

    .. code-block:: python

        bc = df.DirichletBC(self.V, self.phi2, df.DomainBoundary())
        bc.apply(A, b)
        solve(A, self.phi2.vector(), b)

    :math:`\\phi` is now obtained by just adding :math:`\\phi_1` and
    :math:`\\phi_2`,

    .. math::

        \\phi = \\phi_1 + \\phi_2 \\qquad \\qquad (5)

    The demag field is defined as the negative gradient of :math:`\\phi`,
    and is returned by the 'compute_field' function.


    *Arguments (should be this way, isn't yet)*
        V
            a Dolfin VectorFunctionSpace object.
        m
            the Dolfin object representing the (unit) magnetisation
        Ms
            the saturation magnetisation
        unit_length
            the scale of the mesh, defaults to 1.
        project_method
            possible methods are
                * 'magpar'
                * 'project'

    At the moment, we think both methods work for first degree basis
    functions. The 'magpar' method may not work with higher degree
    basis functions, but it is considerably faster than 'project'
    for the kind of problems we are working on now.

    *Example of usage*

        See the exchange_demag example.

    """
    def __init__(self, problem, degree=1, element="CG", project_method='magpar', unit_length=1):
        timings.start("FKSolver init")
        super(FemBemFKSolver, self).__init__(problem, degree, element=element)

        # Solver parameters
        #df.parameters["linear_algebra_backend"] = "PETSc"
        #self.solver = "cg"
        #self.preconditioner = "ilu"
        self.phi1_solver = df.KrylovSolver(self.poisson_matrix)
        self.phi1_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True

        # Data
        self.m = self.M
        self.Ms = problem.Ms
        self.mesh = problem.mesh
        self.unit_length = unit_length
        self.n = df.FacetNormal(self.mesh)
        self.mu0 = np.pi*4e-7 # Vs/(Am)

        # Functions and functionspace (can get a lot of this from base
        # after the interface changes).
        self.W = df.VectorFunctionSpace(self.mesh, element, degree, dim=3)
        self.w = df.TrialFunction(self.W)
        self.vv = df.TestFunction(self.W)
        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)
        self.H_demag = df.Function(self.W)
        self.method = project_method

        # Eq (1) and code-block 2 - two first lines.
        b = self.Ms*df.inner(self.w, df.grad(self.v))*df.dx
        self.D = df.assemble(b)

        # Needed for energy density computation
        self.nodal_vol = df.assemble(self.v*df.dx, mesh=self.mesh).array()
        self.ED = df.Function(self.V)

        if self.method == 'magpar':
            timings.startnext("Setup field magpar method")
            self.__setup_field_magpar()
            self.__compute_field = self.__compute_field_magpar
        elif self.method == 'project':
            self.__compute_field = self.__compute_field_project
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'magpar',
                                    * 'project'""")

        # Compute boundary element matrix and global-to-boundary mapping
        timings.startnext("Build boundary element matrix")
        self.bem, self.b2g_map = compute_bem_fk(OrientedBoundaryMesh(self.mesh))
        timings.stop("Build boundary element matrix")


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

    def compute_field(self):
        """
        Compute the demag field.

        .. note::

            Using this instead of compute_demagfield from base for now.
            The interface has to be changed to this later anyway, so
            we can just keep it this way so we don't need to change the
            examples later.

        *Returns*
            numpy.ndarray
                The demag field.

        """
        self.__solve()
        return self.__compute_field()

    def scalar_potential(self):
        """Return the scalar potential."""
        return self.phi

    def __solve(self):

        # Compute phi1 on the whole domain (code-block 1, last line)

        # Now I'm not sure about the boundary term here..
        timings.start("phi1 - matrix product")
        g1 = self.D*self.m.vector()

        # NOTE: The (above) computation of phi1 is equivalent to
        #timings.start("phi1 - assemble")
        #g1 = df.assemble(self.Ms*df.dot(self.n,self.m)*self.v*df.ds \
        #        - self.Ms*df.div(self.m)*self.v*df.dx)
        # but the way we have implemented it is faster,
        # because we don't have to assemble L each time,
        # and matrix multiplication is faster than assemble.

        timings.startnext("phi1 - solve")
        #df.solve(self.poisson_matrix, self.phi1.vector(), g1, \
        #         self.solver, self.preconditioner)
        self.phi1_solver.solve(self.phi1.vector(), g1)

        # Restrict phi1 to the boundary
        timings.startnext("Restrict phi1 to boundary")
        Phi1 = self.phi1.vector()[self.b2g_map]

        # Compute phi2 on the boundary, eq. (3)
        timings.startnext("Compute phi2 on boundary")
        Phi2 = np.dot(self.bem, Phi1.array())
        self.phi2.vector()[self.b2g_map[:]] = Phi2

        # Compute Laplace's equation inside the domain,
        # eq. (2) and last code-block
        timings.startnext("Compute phi2 inside")
        self.phi2 = self.solve_laplace_inside(self.phi2)

        # phi = phi1 + phi2, eq. (5)
        timings.startnext("Add phi1 and phi2")
        self.phi.vector()[:] = self.phi1.vector() \
                             + self.phi2.vector()
        timings.stop("Add phi1 and phi2")

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
        timings.start("Compute field")
        Hd = self.G*self.phi.vector()
        Hd = Hd.array()/self.L
        timings.stop("Compute field")
        return Hd

    def __compute_field_project(self):
        """
        Dolfin method of projecting the scalar potential
        onto a dolfin.VectorFunctionSpace.

        """
        Hdemag = df.project(-df.grad(self.phi), self.W)
        return Hdemag.vector().array()


if __name__ == "__main__":
    class Problem():
        pass

    problem = Problem()
    mesh = df.UnitSphere(4)
    Ms = 1
    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)

    problem.mesh = mesh
    problem.M = df.project(df.Constant((1, 0, 0)), V)
    problem.Ms = Ms

    demag = FemBemFKSolver(problem)
    Hd = demag.compute_field()
    Hd.shape = (3, -1)
    print np.average(Hd[0])/Ms, np.average(Hd[1])/Ms, np.average(Hd[2])/Ms
    print timings
