"Solvers for the demagnetization field using the Fredkin-Koehler approach"

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

# Modified by Anders E. Johansen, 2012
# Last change: 16.04.2012

from finmag.demag import solver_base as sb
import dolfin as df
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg.dsolve import linsolve
import progressbar as pb
import belement
import belement_magpar
import finmag.util.solid_angle_magpar as solid_angle_solver
compute_belement=belement_magpar.return_bele_magpar()
compute_solid_angle=solid_angle_solver.return_csa_magpar()
import logging
from finmag.util.timings import timings

logger = logging.getLogger(name='finmag')

class FemBemFKSolver(sb.FemBemDeMagSolver):
    def __init__(self, problem, degree=1, element="CG", method='magpar'):
        timings.start("FKSolver init first part")
        super(FemBemFKSolver, self).__init__(problem, degree, element=element)
        self.m = self.M
        self.Ms = problem.Ms
        self.mesh = problem.mesh
        self.W = df.VectorFunctionSpace(self.mesh, element, degree, dim=3)
        self.w = df.TrialFunction(self.W)
        self.vv = df.TestFunction(self.W)
        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)
        self.method = method
        self.nodes_number = self.mesh.num_vertices()
        timings.stop("FKSolver init first part")
        self.bdofs = self.doftionary.keys()

        # Build stuff that doesn't change through time
        self.__build_mapping()
        self.__build_matrices()
        if method == 'magpar':
            self.__compute_volume()

    def compute_field(self):
        """Using this instead of compute_demagfield from base for now."""
        self.__solve()

        if self.method == 'magpar':
            # Magpar method used by Weiwei
            Hdemag = self.G*self.phi.vector()
            return Hdemag.array()/self.L
        elif self.method == 'project':
            Hdemag = df.project(-df.grad(self.phi), self.W)
            return Hdemag.vector().array()
        else:
            raise NotImplementedError("""Only method implemented are
                                    * 'project'
                                    * 'magpar'""")

    def __solve(self):
        # Compute phi1 on the whole domain
        timings.start("Solve for phi1")
        g1 = self.D*self.m.vector()
        df.solve(self.poisson_matrix, self.phi1.vector(), g1)
        timings.stop("Solve for phi1")

        # Restrict phi1 to the boundary
        timings.start("Restrict phi1 to boundary")
        # FIXME: Why doesn't this work?
        # phi1_bnd = self.restrict_to(self.phi1.vector())
        phi1_bnd = self.U1*self.phi1.vector()
        timings.stop("Restrict phi1 to boundary")

        # Compute phi2 on the boundary as a df.dot product
        # between the boundary element matrix and
        # phi1 on the boundary
        timings.start("Compute phi2 on boundary")
        self.phi2_bnd = np.dot(self.bem, phi1_bnd)
        # TODO: Find out if the following line works as it should
        #self.phi2.vector()[self.bdofs[:]] = phi2_bnd[:]
        timings.stop("Compute phi2 on boundary")

        # Compute Laplace's equation inside the domain
        timings.start("Compute phi2 inside")
        # FIXME: Are you correct?
        #self.phi2 = self.solve_laplace_inside(self.phi2)
        self.__do_weiwei_stuff()
        timings.stop("Compute phi2 inside")

        # phi = phi1 + phi2
        timings.start("Add phi1 and phi2")
        self.phi = self.calc_phitot(self.phi1, self.phi2)
        timings.stop("Add phi1 and phi2")

    def __do_weiwei_stuff(self):
        g2 = self.U2*self.phi2_bnd
        self.K2 = self.K2.tocsr()
        phi2 = linsolve.spsolve(self.K2, g2, use_umfpack=False)
        self.phi2.vector()[:] = phi2

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

        timings.start("phi1: build poisson matrix")
        self.build_poisson_matrix()
        timings.stop("phi1: build poisson matrix")

        # Compute boundary element matrix
        timings.start("Build boundary element matrix")
        self.bem = self.__compute_bem()
        timings.stop("Build boundary element matrix")

        # Compute U1 (used to restrict phi1 to boundary)
        timings.start("Compute U1")
        self.U1 = sp.lil_matrix((self.bnd_nodes_number,
                                self.nodes_number),
                                dtype='float32')
        g2b = self.gnodes_to_bnodes
        for i in range(self.nodes_number):
            if g2b[i] >= 0:
                self.U1[g2b[i], i] = 1
        timings.stop("Compute U1")

        ####
        # Temporary code because solve laplace inside may be buggy

        timings.start("Compute U2")
        self.U2 = sp.lil_matrix((self.nodes_number,
                                self.bnd_nodes_number),
                                dtype='float32')

        g2b = self.gnodes_to_bnodes
        tmp_mat = sp.lil_matrix(self.poisson_matrix.array())
        rows,cols = tmp_mat.nonzero()

        for row,col in zip(rows,cols):
            if g2b[row] < 0 and g2b[col] >= 0:
                self.U2[row, g2b[col]] =- tmp_mat[row, col]

        for i in range(self.nodes_number):
            if g2b[i] >= 0:
                self.U2[i, g2b[i]] = 1
        timings.stop("Compute U2")

        timings.start("Compute K2")
        self.K2 = sp.lil_matrix((self.nodes_number,
                                 self.nodes_number),
                                 dtype='float32')

        tmp_mat = sp.lil_matrix(self.poisson_matrix.array())
        rows, cols = tmp_mat.nonzero()
        for row, col in zip(rows, cols):
            if g2b[row] < 0 and g2b[col] < 0:
                self.K2[row, col] = tmp_mat[row, col]

        for i in range(self.nodes_number):
            if g2b[i] >= 0:
                self.K2[i, i] = 1
        timings.stop("Compute K2")


    def __build_mapping(self):
        self.bnd_face_nodes,\
        self.gnodes_to_bnodes,\
        self.bnd_faces_number,\
        self.bnd_nodes_number = \
                belement.compute_bnd_mapping(self.mesh)

    def __compute_volume(self):
        # I have absolutely no idea what this is...
        a = df.inner(df.grad(self.u), self.vv)*df.dx
        self.G = df.assemble(a)

        b = df.dot(self.vv, df.Constant((-1, -1, -1)))*df.dx
        self.L = df.assemble(b)


    def __compute_bem(self):
        mesh = self.mesh
        xyz = mesh.coordinates()
        bfn = self.bnd_face_nodes
        g2b = self.gnodes_to_bnodes

        nodes_number = self.nodes_number

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
        df.solve(a == f, self.phi1, solver_parameters = self.phi1solverparams)

    def solve(self):
        """
        Return the magnetic scalar potential :math:`phi` of
        equation (2.13) in Andreas Knittel's thesis,
        using a FemBem solver and the Fredkin-Koehler approach.
        """

        # Compute phi1 according to Knittel (2.28 - 2.29)
        self.compute_phi1()

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
        phi1 = self.restrict_to(self.phi1.vector())
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

        df.info_blue("Building Boundary Element Matrix")
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

    from finmag.demag.problems.prob_fembem_testcases import MagSphere
    problem = MagSphere(5,1)

    V = df.VectorFunctionSpace(problem.mesh, 'Lagrange', 1)
    Ms = 1e6
    m = df.project(df.Constant((1, 0, 0)), V)

    problem.Ms = Ms
    problem.M = m

    demag = FemBemFKSolver(problem, method='project')
    print demag.compute_field()
    #demag = FemBemFKSolverOld(problem)
    #phi = demag.solve()
    #print demag.get_demagfield(phi).vector().array()
