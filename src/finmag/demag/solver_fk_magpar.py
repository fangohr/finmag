import os
import logging
import numpy as np
import dolfin as df
import scipy.sparse as sp
from finmag.demag import solver_base as sb
from finmag.util.timings import timings
from finmag.util.progress_bar import ProgressBar
import belement
import belement_magpar
import finmag.util.solid_angle_magpar as solid_angle_solver
compute_belement=belement_magpar.return_bele_magpar()
compute_solid_angle=solid_angle_solver.return_csa_magpar()
logger = logging.getLogger(name='finmag')

__all__ = ["FemBemFKSolver"]

class FemBemFKSolver(object):
    def __init__(self, mesh, m, parameters=None, degree=1, element="CG",
                 project_method='magpar', unit_length=1, Ms=1.0):

        self.V = df.FunctionSpace(mesh, element, degree)
        self.W = df.VectorFunctionSpace(mesh, element, degree, dim=3)
        self.u = df.TrialFunction(self.V)
        self.v = df.TestFunction(self.V)
        self.w = df.TrialFunction(self.W)
        self.vv = df.TestFunction(self.W)
        self.phi = df.Function(self.V)
        self.phi1 = df.Function(self.V)
        self.phi2 = df.Function(self.V)
        self.laplace_zeros = df.Function(self.V).vector()


        #Linear Solver parameters
        if parameters:
            p_method = parameters["poisson_solver"]["method"]
            p_pc = parameters["poisson_solver"]["preconditioner"]
            l_method = parameters["laplace_solver"]["method"]
            l_pc = parameters["laplace_solver"]["preconditioner"]
        else:
            p_method, p_pc, l_method, l_pc = "default", "default", "default", "default"

        self.poisson_matrix = df.assemble(df.inner(df.grad(self.u), df.grad(self.v))*df.dx, mesh=mesh)
        self.poisson_solver = df.KrylovSolver(self.poisson_matrix, p_method, p_pc)
        self.laplace_solver = df.KrylovSolver(l_method, l_pc)

        # Eq (1) and code-block 2 - two first lines.
        b = Ms*df.inner(self.w, df.grad(self.v))*df.dx
        self.D = df.assemble(b)

        self.m = m
        self.Ms = Ms
        self.mesh = mesh
        self.build_all()
        #self.solve()

    def solve(self):
        timings.start("phi1 - matrix product")
        b = self.D*self.m.vector()
        #b = df.assemble(self.Ms*df.dot(self.n,self.m)*self.v*df.ds \
        #        - self.Ms*df.div(self.m)*self.v*df.dx)
        timings.startnext("phi1 - solve")
        self.poisson_solver.solve(self.phi1.vector(), b)

        timings.startnext("Restrict phi1 to boundary")
        Phi1 = self.U1*self.phi1.vector().array()

        timings.startnext("Compute Phi2")
        Phi2 = np.dot(self.bem, Phi1)

        timings.startnext("phi2 <- Phi2")
        self.phi2.vector()[:] = self.U2*Phi2

        timings.startnext("Compute phi2 inside")
        bc = df.DirichletBC(self.V, self.phi2, df.DomainBoundary())
        A = self.poisson_matrix.copy()
        b = self.laplace_zeros.copy()
        bc.apply(A, b)
        self.laplace_solver.solve(A, self.phi2.vector(), b)

        timings.startnext("Add phi1 and phi2")
        self.phi.vector()[:] = self.phi1.vector() \
                             + self.phi2.vector()
        timings.stop("Add phi1 and phi2")

    def compute_field(self):
        self.solve()
        Hd = self.G*self.phi.vector()
        Hd = Hd.array()/self.L
        return Hd

    def build_all(self):
        # Mapping
        self.bnd_face_nodes,\
        self.gnodes_to_bnodes,\
        self.bnd_faces_number,\
        self.bnd_nodes_number= \
                belement.compute_bnd_mapping(self.mesh)
        self.nodes_number=self.mesh.num_vertices()

        # Project
        a = df.inner(df.grad(self.u), self.vv)*df.dx
        b = df.dot(self.vv, df.Constant([-1, -1, -1]))*df.dx
        self.G = df.assemble(a)
        self.L = df.assemble(b).array()

        # U1
        self.U1 = sp.lil_matrix((self.bnd_nodes_number,
                                 self.nodes_number),
                                 dtype='float32')
        g2b=self.gnodes_to_bnodes
        for i in range(self.nodes_number):
            if g2b[i]>=0:
                self.U1[g2b[i],i]=1

        # U2
        self.U2 = sp.lil_matrix((self.nodes_number,
                                 self.bnd_nodes_number),
                                 dtype='float32')
        self.K1 = self.poisson_matrix
        g2b=self.gnodes_to_bnodes
        tmp_mat=sp.lil_matrix(self.K1.array())
        rows,cols = tmp_mat.nonzero()
        for row,col in zip(rows,cols):
            if g2b[row]<0 and g2b[col]>=0:
                self.U2[row,g2b[col]]=-tmp_mat[row,col]
        for i in range(self.nodes_number):
            if g2b[i]>=0:
                self.U2[i,g2b[i]]=1

        # BEM
        self.compute_BEM_matrix()


    def compute_BEM_matrix(self):
        mesh=self.mesh
        xyz=mesh.coordinates()
        bfn=self.bnd_face_nodes
        g2b=self.gnodes_to_bnodes

        nodes_number=mesh.num_vertices()
        n=self.bnd_nodes_number
        B=np.zeros((n,n))

        tmp_bele=np.array([0.,0.,0.])
        loops = (nodes_number - sum(g2b<0))*self.bnd_faces_number + mesh.num_cells()*4
        loop_ctr = 0
        bar = ProgressBar(loops)
        logger.info("Building Boundary Element Matrix")

        for i in range(nodes_number):
            #skip the node not at the boundary
            if g2b[i]<0:
                continue
            for j in range(self.bnd_faces_number):

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
        vert_bsa=np.zeros(nodes_number)

        mc=mesh.cells()
        for i in range(mesh.num_cells()):
            for j in range(4):
                tmp_omega=compute_solid_angle(
                    xyz[mc[i][j]],
                    xyz[mc[i][(j+1)%4]],
                    xyz[mc[i][(j+2)%4]],
                    xyz[mc[i][(j+3)%4]])
                vert_bsa[mc[i][j]]+=tmp_omega

                loop_ctr += 1
                bar.update(loop_ctr)

        for i in range(nodes_number):
            j=g2b[i]
            if j<0:
                continue
            B[j][j]+=vert_bsa[i]/(4*np.pi)-1


        self.bem=B



class FemBemFKSolverORG(sb.FemBemDeMagSolver):
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
    This is given by Knittel's thesis, eq. (2.27) - (2.29).

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

    These three equations were taken from Knittel's thesis, equations
    (2.30) - (2.32)

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
    :math:`\mathbf{B}` are given by Knittel (2.52):

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
    def __init__(self, mesh,m, parameters=None, degree=1, element="CG",
                 project_method='magpar', unit_length=1,Ms = 1.0):

        timings.start("FKSolver init")
        sb.FemBemDeMagSolver.__init__(self,mesh,m, parameters, degree, element=element,
                                      project_method = project_method,
                                      unit_length = unit_length,Ms = Ms)

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
        #self.bem, self.b2g_map = compute_bem_fk(OrientedBoundaryMesh(self.mesh))
        self.build_all()
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
        #Phi1 = self.phi1.vector()[self.b2g_map]
        Phi1 = self.U1*self.phi1.vector().array()

        # Compute phi2 on the boundary, eq. (3)
        timings.startnext("Compute Phi2")
        Phi2 = np.dot(self.bem, Phi1)

        # Fill Phi2 into boundary positions of phi2
        timings.startnext("phi2 <- Phi2")
        #self.phi2.vector()[self.b2g_map[:]] = Phi2
        self.phi2.vector()[:] = self.U2*Phi2

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

    def build_all(self):
        # Mapping
        self.bnd_face_nodes,\
        self.gnodes_to_bnodes,\
        self.bnd_faces_number,\
        self.bnd_nodes_number= \
                belement.compute_bnd_mapping(self.mesh)
        self.nodes_number=self.mesh.num_vertices()

        # BEM
        self.compute_BEM_matrix()

        # U1
        self.U1 = sp.lil_matrix((self.bnd_nodes_number,
                                 self.nodes_number),
                                 dtype='float32')
        g2b=self.gnodes_to_bnodes
        for i in range(self.nodes_number):
            if g2b[i]>=0:
                self.U1[g2b[i],i]=1

        # U2
        self.U2 = sp.lil_matrix((self.nodes_number,
                                 self.bnd_nodes_number),
                                 dtype='float32')
        self.K1 = self.poisson_matrix
        g2b=self.gnodes_to_bnodes
        tmp_mat=sp.lil_matrix(self.K1.array())
        rows,cols = tmp_mat.nonzero()
        for row,col in zip(rows,cols):
            if g2b[row]<0 and g2b[col]>=0:
                self.U2[row,g2b[col]]=-tmp_mat[row,col]
        for i in range(self.nodes_number):
            if g2b[i]>=0:
                self.U2[i,g2b[i]]=1


    def compute_BEM_matrix(self):
        mesh=self.mesh
        xyz=mesh.coordinates()
        bfn=self.bnd_face_nodes
        g2b=self.gnodes_to_bnodes

        nodes_number=mesh.num_vertices()
        n=self.bnd_nodes_number
        B=np.zeros((n,n))

        tmp_bele=np.array([0.,0.,0.])
        loops = (nodes_number - sum(g2b<0))*self.bnd_faces_number + mesh.num_cells()*4
        loop_ctr = 0
        bar = ProgressBar(loops)
        logger.info("Building Boundary Element Matrix")

        for i in range(nodes_number):
            #skip the node not at the boundary
            if g2b[i]<0:
                continue
            for j in range(self.bnd_faces_number):

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
        vert_bsa=np.zeros(nodes_number)

        mc=mesh.cells()
        for i in range(mesh.num_cells()):
            for j in range(4):
                tmp_omega=compute_solid_angle(
                    xyz[mc[i][j]],
                    xyz[mc[i][(j+1)%4]],
                    xyz[mc[i][(j+2)%4]],
                    xyz[mc[i][(j+3)%4]])
                vert_bsa[mc[i][j]]+=tmp_omega

                loop_ctr += 1
                bar.update(loop_ctr)

        for i in range(nodes_number):
            j=g2b[i]
            if j<0:
                continue
            B[j][j]+=vert_bsa[i]/(4*np.pi)-1


        self.bem=B


if __name__ == "__main__":
    from finmag.demag.problems import prob_fembem_testcases as pft
    problem = pft.MagSphere20()
    Ms = problem.Ms
    
    demag = FemBemFKSolver(**problem.kwargs())
    Hd = demag.compute_field()
    Hd.shape = (3, -1)
    print np.average(Hd[0])/Ms, np.average(Hd[1])/Ms, np.average(Hd[2])/Ms
    print timings
    df.plot(demag.phi)
    df.interactive()
