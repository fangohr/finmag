from finmag.demag import solver_base as sb
import dolfin as df
import numpy as np
import progressbar as pb
import belement
import belement_magpar
import finmag.util.solid_angle_magpar as solid_angle_solver
compute_belement=belement_magpar.return_bele_magpar()
compute_solid_angle=solid_angle_solver.return_csa_magpar()
import logging
from finmag.util.timings import timings

logger = logging.getLogger(name='finmag')

__all__ = ["FemBemFKSolver"]
class FemBemFKSolver(sb.FemBemDeMagSolver):
    """
    The idea of the Fredkin-Koehler approach is to split the magnetic
    potential into two parts, :math:`\\phi = \\phi_1 + \\phi_2`.

    :math:`\\phi_1` solves the inhomogeneous Neumann problem

    .. math::

        \\Delta \\phi_1 = \\nabla \\cdot \\vec M(\\vec r), \\quad \\vec r \\in \\Omega,

    with

    .. math::

        \\frac{\partial \\phi_1}{\\partial \\vec n} = \\vec n \\cdot \\vec M

    on :math:`\\Gamma`. In addition, :math:`\\phi_1(\\vec r) = 0` for
    :math:`\\vec r \\not \\in \\Omega`.

    Multiplying with a test function, :math:`v`, and integrate over the domain,
    the corresponding variational problem reads

    .. math::

        \\int_\\Omega \\nabla \\phi_1 \\cdot \\nabla v =
        \\int_\\Omega (\\nabla \\cdot \\vec
        M)v \\mathrm{d}x

    This could be solved straight forward by

    .. code-block:: python

        a = df.inner(df.grad(u), df.grad(v))*df.dx
        L = self.Ms*df.div(m)*self.v*df.dx
        df.solve(a==L, self.phi1)

    but we are instead using the fact that L can be written as

    .. code-block:: python

        b = Ms*df.inner(w, df.grad(v))*df.dx
        D = df.assemble(b)
        L = D*m.vector()

    In this way, we can assemble D at setup, and do not have to
    recompute it each time. This speeds up the solver significantly.

    :math:`\\phi_2` is the solution of Laplace's equation inside the domain,

    .. math::

        \\Delta \\phi_2(\\vec r) = 0
        \\quad \\hbox{for } \\vec r \\in \\Omega.

    At the boundary, :math:`\\phi_2` has a discontinuity of

    .. math::

        \\bigtriangleup \\phi2(\\vec r) = \\phi1(\\vec r),

    and it disappears at infinity, i.e.

    .. math::

        \\phi_2(\\vec r) \\rightarrow 0 \\quad \\mathrm{for}
        \\quad \\lvert \\vec r \\rvert \\rightarrow \\infty.

    In contrast to the Poisson equation for :math:`\\phi_1`,
    which is solved straight forward in a finite domain, we now need to
    apply a BEM technique to solve the equations for :math:`\\phi_2`.
    First, we solve the equation on the boundary. By eq. (2.51) in Knittel's
    thesis, this yieds

    .. math::

        \\Phi_2 = \\mathbf{B} \\cdot \\Phi_1,

    with :math:`\\Phi_1` as the vector of elements from :math:`\\phi_1` which
    is on the boundary. These are found by the 'restrict_to' function written
    in C-code.

    .. code-block:: python

        Phi1 = self.restrict_to(self.phi1.vector())

    The elements of the boundary element matrix
    :math:`\\mathbf{B}` given by

    .. math::

        B_{ij} = \\frac{1}{4\\pi}\\int_{\\Gamma_j} \\psi_j(\\vec r)
        \\frac{(\\vec R_i - \\vec r) \\cdot n(\\vec r)}
        {\\lvert \\vec R_i - \\vec r \\rvert^3} \\mathrm{d}s +
        \\left(\\frac{\\Omega(\\vec R_i)}{4\\pi} - 1 \\right) \\delta_{ij}.

    Here, :math:`\\psi` is a set of basis functions and
    :math:`\\Omega(\\vec R)` denotes the solid angle.

    .. note::

        When our own boundary element matrix is implemented, write
        something about it here.

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

        self.phi2.vector()[self.bdofs[:]] = self.Phi2[:]

    And this can now be applied to DirichletBC to create boundary
    conditions. Remember that A is our previously assembled Poisson matrix,
    and b is here a zero vector. The complete code then reads

    .. code-block:: python

        bc = df.DirichletBC(self.V, self.phi2, df.DomainBoundary())
        bc.apply(A, b)
        solve(A, self.phi2.vector(), b)

    :math:`\\phi` is now obtained by just adding :math:`\\phi_1` and
    :math:`\\phi_2`.

    The demag field is defined as the negative gradient of :math:`\\phi`,
    and is returned by the 'compute_field' function.


    *Arguments (should be this way, isn't yet)*
        V
            a Dolfin VectorFunctionSpace object.
        m
            the Dolfin object representing the (unit) magnetisation
        Ms
            the saturation magnetisation
        method
            possible methods are
                * 'magpar'
                * 'project'

    At the moment, we think both methods work for first degree basis
    functions. The 'magpar' method may not work with higher degree
    basis functions, but it is considerably faster than 'project'
    for the kind of problems we are working with now.

    *Example of usage*

        See the exchange_demag example.

    """
    def __init__(self, problem, degree=1, element="CG", method='magpar'):
        timings.start("FKSolver init first part")
        super(FemBemFKSolver, self).__init__(problem, degree, element=element)

        # Data
        self.m = self.M
        self.Ms = problem.Ms
        self.mesh = problem.mesh

        # Functions and functionspace (can get a lot of this from base
        # after the interface changes.
        self.W = df.VectorFunctionSpace(self.mesh, element, degree, dim=3)
        self.w = df.TrialFunction(self.W)
        self.vv = df.TestFunction(self.W)
        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)
        self.laplace_zeros = df.Function(self.V).vector()
        self.method = method
        timings.stop("FKSolver init first part")

        # Build stuff that doesn't change through time
        self.__build_matrices()
        if method == 'magpar':
            self.__setup_field_magpar()
            self.__compute_field = self.__compute_field_magpar
        elif method == 'project':
            self.__compute_field = self.__compute_field_project
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'magpar',
                                    * 'project'""")

        # I think these are the fastest solvers I could find today.
        self.phi1_solver = df.KrylovSolver(self.poisson_matrix)
        #self.phi2_solver = df.KrylovSolver()
        self.phi1_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True
        #self.phi2_solver.parameters["preconditioner"]["same_nonzero_pattern"] = True

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

    def __solve(self):
        # Compute phi1 on the whole domain
        timings.start("phi1 - product")
        g1 = self.D*self.m.vector()
        timings.startnext("phi1 - solve")
        self.phi1_solver.solve(self.phi1.vector(), g1)
        # NOTE: The computation of phi1 is equivalent to
        #
        #a = df.inner(df.grad(self.u), df.grad(self.v))*df.dx
        #L = self.Ms*df.div(self.m)*self.v*df.dx
        #df.solve(a==L, self.phi1)
        #
        # but the way we have implemented it is faster,
        # because we don't have to assemble L each time.

        # Restrict phi1 to the boundary
        timings.startnext("Restrict phi1 to boundary")
        Phi1 = self.restrict_to(self.phi1.vector())

        # I can't explain why this seems to work
        #U1 = df.PETScMatrix()
        #U1.resize(self.bnd_nodes_number, self.nodes_number)
        #U1.ident_zeros()
        #Phi1 = U1*self.phi1.vector()

        # Compute phi2 on the boundary as a dot product
        # between the boundary element matrix and
        # phi1 on the boundary
        timings.startnext("Compute phi2 on boundary")
        self.Phi2 = np.dot(self.bem, Phi1)
        self.phi2.vector()[self.bdofs[:]] = self.Phi2[:]

        # Compute Laplace's equation inside the domain
        timings.startnext("Compute phi2 inside")
        self.phi2 = self.solve_laplace_inside(self.phi2)
        #self.solve_laplace_inside()
        # phi = phi1 + phi2
        timings.startnext("Add phi1 and phi2")
        self.phi = self.calc_phitot(self.phi1, self.phi2)
        timings.stop("Add phi1 and phi2")

    '''
    def solve_laplace_inside(self):
        """Suspect this function from solver_base to be buggy."""
        #NOTE: This is now fixed, but keep this until we are really
        #sure that it's fixed.
        bc = df.DirichletBC(self.V, self.phi2, df.DomainBoundary())
        A = self.poisson_matrix.copy()
        b = self.laplace_zeros.copy()
        bc.apply(A, b)
        self.phi2_solver.solve(A, self.phi2.vector(), b)
    '''

    def __build_matrices(self):
        """
        Build anything that doesn't depend on m to avoid
        things being build multiple times.
        """
        Ms, m, u, v, w = self.Ms, self.m, self.u, self.v, self.w

        # phi1 is the solution of poisson_matrix * phi1 = D*m
        timings.start("phi1: compute D")
        b = Ms*df.inner(w, df.grad(v))*df.dx
        self.D = df.assemble(b)
        timings.stop("phi1: compute D")

        # Compute boundary element matrix
        timings.start("Build boundary element matrix")
        self.bem = self.__compute_bem()
        timings.stop("Build boundary element matrix")

    def __setup_field_magpar(self):
        """Needed by the magpar method we may use instead of project."""
        timings.start("Setup field magpar method")
        a = df.inner(df.grad(self.u), self.vv)*df.dx
        self.G = df.assemble(a)

        b = df.dot(self.vv, df.Constant([-1, -1, -1]))*df.dx
        self.L = df.assemble(b).array()
        timings.stop("Setup field magpar method")

    def __compute_field_magpar(self):
        """Magpar method used by Weiwei."""
        timings.start("G")
        Hd = self.G*self.phi.vector()
        timings.startnext("L")
        Hd = Hd.array()/self.L
        timings.stop("L")
        return Hd

    def __compute_field_project(self):
        timings.start("Project")
        Hdemag = df.project(-df.grad(self.phi), self.W)
        timings.stop("Project")
        return Hdemag.vector().array()

    def __build_mapping(self):
        """
        Only used by Weiwei's magpar code to
        compute the boundary element matrix.
        """
        self.bnd_face_nodes,\
        self.gnodes_to_bnodes,\
        self.bnd_faces_number,\
        self.bnd_nodes_number = \
                belement.compute_bnd_mapping(self.mesh)

    def __compute_bem(self):
        """
        Code written by Weiwei to compute the boundary
        element matrix using magpar code.
        """
        self.__build_mapping()
        mesh = self.mesh
        xyz = mesh.coordinates()
        bfn = self.bnd_face_nodes
        g2b = self.gnodes_to_bnodes

        nodes_number = self.mesh.num_vertices()

        n = self.bnd_nodes_number
        B = np.zeros((n,n))

        tmp_bele = np.zeros(3)

        # Progressbar..
        loops = (nodes_number - sum(g2b<0))*len(bfn) + mesh.num_cells()*4
        loop_ctr = 0
        bar = pb.ProgressBar(maxval=loops, \
                widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
        logger.info("Building Boundary Element Matrix")

        for i in range(nodes_number):
            #skip the node not at the boundary
            if g2b[i]<0:
                continue

            for j in range(self.bnd_faces_number):

                # Progressbar data
                loop_ctr += 1
                bar.update(loop_ctr)

                #skip the node in the face
                if i in set(bfn[j]):
                    continue

                compute_belement(
                    xyz[i],
                    xyz[bfn[j][0]],
                    xyz[bfn[j][1]],
                    xyz[bfn[j][2]],
                tmp_bele)

                for k in range(3):
                    ti=g2b[i]
                    tj=g2b[bfn[j][k]]
                    B[ti][tj]+=tmp_bele[k]


        #the solid angle term ...
        solid_angle = np.zeros(nodes_number)

        cells = mesh.cells()
        for i in range(mesh.num_cells()):
            for j in range(4):
                omega = compute_solid_angle(
                    xyz[cells[i][j]],
                    xyz[cells[i][(j+1)%4]],
                    xyz[cells[i][(j+2)%4]],
                    xyz[cells[i][(j+3)%4]])

                solid_angle[cells[i][j]] += omega

                # Progressbar data
                loop_ctr += 1
                bar.update(loop_ctr)

        for i in range(nodes_number):
            j = g2b[i]
            if j < 0:
                continue

            B[j][j] += solid_angle[i]/(4*np.pi) - 1

        return B



class FemBemFKSolverOld(sb.FemBemDeMagSolver):
    # Very deprecated, should be removed soon.
    """FemBem solver for Demag Problems using the Fredkin-Koehler approach."""

    def __init__(self, problem, degree=1):
        super(FemBemFKSolverOld,self).__init__(problem, degree)
        #Default linalg solver parameters
        self.phi1solverparams = {"linear_solver":"lu"}
        self.phi2solverparams = {"linear_solver":"lu"}
        self.degree = degree
        self.mesh = problem.mesh
        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)

    def compute_phi1(self):
        """
        Get phi1 defined over the mesh with the point given the value 0.
        phi1 is the solution to the inhomogeneous Neumann problem.
        M is the magnetisation field.
        V is a df.FunctionSpace

        """
        # Define functions
        n = df.FacetNormal(self.mesh)

        # Define forms
        eps = 1e-8
        a = df.dot(df.grad(self.u), df.grad(self.v))*df.dx - \
                df.dot(eps*self.u, self.v)*df.dx
        f = df.dot(n, self.M)*self.v*df.ds - df.div(self.M)*self.v*df.dx
        self.linsolve_phi1(a,f)

    def linsolve_phi1(self,a,f):
        # Solve for the DOFs in phi1
        df.solve(a == f, self.phi1)#, solver_parameters = self.phi1solverparams)

    def solve(self):
        """
        Return the magnetic scalar potential :math:`phi` of
        equation (2.13) in Andreas Knittel's thesis,
        using a FemBem solver and the Fredkin-Koehler approach.
        """

        # Compute phi1 according to Knittel (2.28 - 2.29)
        print "Presolve:", type(self.phi1.vector())
        self.compute_phi1()
        print "Postsolve:", type(self.phi1.vector())
        # Compute phi2 on the boundary
        self.__solve_phi2_boundary(self.doftionary)

        # Compute Laplace's equation (Knittel, 2.30)
        self.solve_laplace_inside(self.phi2, self.phi2solverparams)

        # Knittel (2.27), phi = phi1 + phi2
        self.phi = self.calc_phitot(self.phi1, self.phi2)

        # Compute the demag field from phi
        self.Hdemag = self.get_demagfield(self.phi)

        return self.phi

    def __solve_phi2_boundary(self, doftionary):
        """Compute phi2 on the boundary."""
        B = self.__build_BEM_matrix(doftionary)
        print "Pre-restrict:", type(self.phi1.vector())
        phi1 = self.restrict_to(self.phi1.vector())
        exit()
        phi2dofs = np.dot(B, phi1)
        bdofs = doftionary.keys()
        for i in range(len(bdofs)):
            self.phi2.vector()[bdofs[i]] = phi2dofs[i]

    def __build_BEM_matrix(self, doftionary):
        """Build and return the Boundary Element Matrix."""
        n = len(doftionary)
        keys = doftionary.keys()
        BEM = np.zeros((n,n))

        bar = pb.ProgressBar(maxval=n-1, \
                widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])

        #df.info_blue("Building Boundary Element Matrix")
        for i, dof in enumerate(doftionary):
            bar.update(i)
            BEM[i] = self.__get_BEM_row(doftionary[dof], keys, i)
        return BEM

    def __get_BEM_row(self, R, dofs, i):
        """Return row i in the BEM."""
        v = self.__BEMnumerator(R)
        w = self.__BEMdenominator(R)
        psi = df.TestFunction(self.V)

        # First term containing integration part (Knittel, eq. (2.52))
        L = 1.0/(4*df.DOLFIN_PI)*psi*v*w*df.ds

        # Create a row from the first term
        bigrow = df.assemble(L, form_compiler_parameters=self.ffc_options)
        row = self.restrict_to(bigrow)

        # Previous implementation of solid angle is proven wrong.
        # Positive thing is that when implemented correctly,
        # the solver will probably run faster.
        #
        # TODO: Implement solid angle.
        SA = self.__solid_angle()

        # Insert second term containing solid angle
        row[i] += SA/(4*np.pi) - 1

        return row

    def __BEMnumerator(self, R):
        """
        Compute the numerator of the fraction in
        the first term in Knittel (2.52)

        """
        dim = len(R)
        v = []
        for i in range(dim):
            v.append("R%d - x[%d]" % (i, i))
        v = tuple(v)

        kwargs = {"R" + str(i): R[i] for i in range(dim)}
        v = df.Expression(v, **kwargs)

        W = df.VectorFunctionSpace(self.problem.mesh, "CR", self.degree, dim=dim)
        v = df.interpolate(v, W)
        n = df.FacetNormal(self.V.mesh())
        v = df.dot(v, n)
        return v

    def __BEMdenominator(self, R):
        """
        Compute the denominator of the fraction in
        the first term in Knittel (2.52)

        """
        w = "pow(1.0/sqrt("
        dim = len(R)
        for i in range(dim):
            w += "(R%d - x[%d])*(R%d - x[%d])" % (i, i, i, i)
            if not i == dim-1:
                w += "+"
        w += "), 3)"

        kwargs = {"R" + str(i): R[i] for i in range(dim)}
        return df.Expression(w, **kwargs)

    def __solid_angle(self):
        """
        TODO: Write solid angle function here.

        """
        SA = 0

        # Solid angle must not exceed the unit circle
        assert abs(SA) <= 2*np.pi

        return SA

if __name__ == "__main__":

    #from finmag.demag.problems.prob_fembem_testcases import MagSphere
    #problem = MagSphere(5,1.5)

    from finmag.demag.problems.prob_base import FemBemDeMagProblem
    from finmag.util.convert_mesh import convert_mesh
    #mesh = df.Mesh("../../../examples/exchange_demag/bar30_30_100.xml.gz")
    #mesh = df.Box(0,0,0,30,30,100,3,3,10)
    mesh = df.UnitSphere(4)
    V = df.VectorFunctionSpace(mesh, 'Lagrange', 1)
    Ms = 1e6
    m = df.project(df.Constant((1, 0, 0)), V)
    problem = FemBemDeMagProblem(mesh, m)

    problem.Ms = Ms
    #problem.M = m

    demag = FemBemFKSolver(problem)
    Hd = demag.compute_field()
    Hd.shape = (3, -1)
    print np.average(Hd[0])/Ms, np.average(Hd[1])/Ms, np.average(Hd[2])/Ms
    #demag = FemBemFKSolverOld(problem)
    #phi = demag.solve()
    #print demag.get_demagfield(phi).vector().array()
