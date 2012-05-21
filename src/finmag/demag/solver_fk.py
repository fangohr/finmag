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
    r"""
    The idea of the Fredkin-Koehler approach is to split the magnetic
    potential into two parts, :math:`\phi = \phi_1 + \phi_2`.

    :math:`\phi_1` solves the inhomogeneous Neumann problem

    .. math::

        \Delta \phi_1 = \nabla \cdot \vec M(\vec r), \quad \vec r \in \Omega, \qquad
        \qquad

    with

    .. math::

        \frac{\partial \phi_1}{\partial \vec n} = \vec n \cdot \vec M \qquad \qquad

    on :math:`\Gamma`. In addition, :math:`\phi_1(\vec r) = 0` for
    :math:`\vec r \not \in \Omega`.

    Multiplying with a test function, :math:`v`, and integrate over the domain,
    we obtain

    .. math::

        \int_\Omega \Delta \phi_1 v \mathrm{d}x = \int_\Omega (\nabla \cdot \vec
        M)v \mathrm{d}x.

    Integration by parts on the laplace term gives

    .. math::

        \int_\Omega \Delta \phi_1 v \mathrm{d}x = \int_{\partial \Omega}
        \frac{\partial \phi_1}{\partial \vec n} v \mathrm{d}s -
        \int_\Omega \nabla \phi_1 \cdot \nabla v \mathrm{d}x.

    Hence our variational problem reads

    .. math::

        \int_\Omega \nabla \phi_1 \cdot \nabla v \mathrm{d}x =
        \int_{\partial \Omega} (\vec n \cdot \vec M) v \mathrm{d}s -
        \int_\Omega (\nabla \cdot \vec
        M)v \mathrm{d}x, \qquad \qquad (1)

    because :math:`\frac{\partial \phi_1}{\partial \vec n} = \vec n \cdot \vec M`.
    This could be solved straight forward by (code-block 1)

    .. code-block:: python

        a = df.inner(df.grad(u), df.grad(v))*df.dx
        L = self.Ms*df.inner(self.n, self.m)*self.v*df.ds - Ms*df.div(m)*self.v*df.dx
        df.solve(a==L, self.phi1)

    but we are instead using that L can be written as (code-block 2)

    .. code-block:: python

        b = Ms*df.inner(w, df.grad(v))*df.dx
        D = df.assemble(b)
        L = D*m.vector()

    What we have used here, is the fact that by integration by parts on the
    divergence term in (1), we get the
    same boundary integral as before, just with different signs,
    so the boundary terms vanish. Proof:

    .. math::

        \int_\Omega \nabla \phi_1 \cdot \nabla v
        &= \int_{\partial \Omega} (\vec n \cdot \vec M) v \mathrm{d}s
        - \int_\Omega (\nabla \cdot \vec M)v \mathrm{d}x \\
        &= \int_{\partial \Omega} (\vec n \cdot \vec M) v \mathrm{d}s
        - \int_{\partial \Omega} (\vec n \cdot \vec M) v \mathrm{d}s
        + \int_\Omega \vec M \cdot \nabla v \mathrm{d}x \\
        &= \int_\Omega \vec M \cdot \nabla v \mathrm{d}x

    The first equality is from (1), and the second in integration by parts
    on the divergence term, using Gauss' divergence theorem.

    Now, we can substitute :math:`\vec M` by a trial function (w in
    code-block 2) defined on the same function space,
    assemble the form, and then multiply with :math:`\vec M`.
    This way, we can assemble D (from code-block 2) at setup,
    and do not have to recompute it each time.
    This speeds up the solver significantly.

    :math:`\phi_2` is the solution of Laplace's equation inside the domain,

    .. math::

        \Delta \phi_2(\vec r) = 0
        \quad \hbox{for } \vec r \in \Omega. \qquad \qquad (2)

    At the boundary, :math:`\phi_2` has a discontinuity of

    .. math::

        \bigtriangleup \phi_2(\vec r) = \phi_1(\vec r), \qquad \qquad

    and it disappears at infinity, i.e.

    .. math::

        \phi_2(\vec r) \rightarrow 0 \quad \mathrm{for}
        \quad \lvert \vec r \rvert \rightarrow \infty. \qquad \qquad

    In contrast to the Poisson equation for :math:`\phi_1`,
    which is solved straight forward in a finite domain, we now need to
    apply a BEM technique to solve the equations for :math:`\phi_2`.
    First, we solve the equation on the boundary. By eq. (2.51) in Knittel's
    thesis, this yieds

    .. math::

        \Phi_2 = \mathbf{B} \cdot \Phi_1, \qquad \qquad (3)

    with :math:`\Phi_1` as the vector of elements from :math:`\phi_1` which
    is on the boundary. These are found by the the global-to-boundary mapping

    .. code-block:: python

        Phi1 = self.phi1.vector().array()[g2b_map]

    The elements of the boundary element matrix
    :math:`\mathbf{B}` are given by

    .. math::

        B_{ij} = \frac{1}{4\pi}\int_{\Gamma_j} \psi_j(\vec r)
        \frac{(\vec R_i - \vec r) \cdot n(\vec r)}
        {\lvert \vec R_i - \vec r \rvert^3} \mathrm{d}s +
        \left(\frac{\Omega(\vec R_i)}{4\pi} - 1 \right) \delta_{ij}. \qquad \qquad (4)

    Here, :math:`\psi` is a set of basis functions and
    :math:`\Omega(\vec R)` denotes the solid angle.

    Having both :math:`\Phi_1` and :math:`\mathbf{B}`,
    we use numpy.dot to compute the dot product.

    .. code-block:: python

        self.Phi2 = np.dot(self.bem, Phi1)

    Now that we have obtained the values of :math:`\phi_2` on the boundary,
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

    :math:`\phi` is now obtained by just adding :math:`\phi_1` and
    :math:`\phi_2`,

    .. math::

        \phi = \phi_1 + \phi_2 \qquad \qquad (5)

    The demag field is defined as the negative gradient of :math:`\phi`,
    and is returned by the 'compute_field' function.

    *For an interface more inline with the rest of FinMag Code please use
    the wrapper class Demag in finmag/energies/demag.*

    *Arguments*
        problem
            An object of type DemagProblem
        parameters
            dolfin.Parameters of method and preconditioner to linear solvers.
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

    At the moment, we think both methods work for first degree basis
    functions. The 'magpar' method may not work with higher degree
    basis functions, but it is considerably faster than 'project'
    for the kind of problems we are working on now.

    *Example of usage*

        See the exchange_demag example.

    """
    def __init__(self, problem, parameters=None, degree=1, element="CG",
                 project_method='magpar', unit_length=1):

        timings.start("FKSolver init")
        sb.FemBemDeMagSolver.__init__(self,problem, parameters, degree, element=element,
                                      project_method = project_method,
                                      unit_length = unit_length)

        #Linear Solver parameters
        if parameters:
            method = parameters["poisson_solver"]["method"]
            pc = parameters["poisson_solver"]["preconditioner"]
        else:
            method, pc = "default", "default"

        self.poisson_solver = df.KrylovSolver(self.poisson_matrix, method, pc)

        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)

        # Eq (1) and code-block 2 - two first lines.
        b = self.Ms*df.inner(self.w, df.grad(self.v))*df.dx
        self.D = df.assemble(b)

        # Compute boundary element matrix and global-to-boundary mapping
        timings.startnext("Build boundary element matrix")
        self.bem, self.b2g_map = compute_bem_fk(OrientedBoundaryMesh(self.mesh))
        timings.stop("Build boundary element matrix")

    def solve(self):

        # Compute phi1 on the whole domain (code-block 1, last line)
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
        self.poisson_solver.solve(self.phi1.vector(), g1)

        # Restrict phi1 to the boundary
        timings.startnext("Restrict phi1 to boundary")
        Phi1 = self.phi1.vector()[self.b2g_map]

        # Compute phi2 on the boundary, eq. (3)
        timings.startnext("Compute Phi2")
        Phi2 = np.dot(self.bem, Phi1.array())

        # Fill Phi2 into boundary positions of phi2
        timings.startnext("phi2 <- Phi2")
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
        return self.phi

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
