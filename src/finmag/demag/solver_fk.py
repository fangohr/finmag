"Solvers for the demagnetization field using the Fredkin-Koehler approach" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

# Modified by Anders E. Johansen, 2012
# Last change: 15.03.2012

from dolfin import *
from finmag.demag import solver_base as sb
import numpy as np
import progressbar as pb

class FKSolver(sb.DeMagSolver):
    """Class containing methods shared by FK solvers"""
    def __init__(self, problem, degree=1):
        super(FKSolver, self).__init__(problem, degree)
        self.phi1 = Function(self.V)
        self.phi2 = Function(self.V)
        self.degree = degree

    def compute_phi1(self, M, V):
        """
        Get phi1 defined over the mesh with the point given the value 0.
        phi1 is the solution to the inhomogeneous Neumann problem.
        M is the magnetisation field.
        V is a FunctionSpace

        """
        # Define functions
        u = TrialFunction(V)
        v = TestFunction(V)
        n = FacetNormal(V.mesh())
        
        # Define forms
        eps = 1e-8
        a = dot(grad(u),grad(v))*dx - dot(eps*u,v)*dx 
        f = dot(n, M)*v*ds - div(M)*v*dx

        # Solve for the DOFs in phi1
        solve(a == f, self.phi1)


class FemBemFKSolver(FKSolver, sb.FemBemDeMagSolver):
    """FemBem solver for Demag Problems using the Fredkin-Koehler approach."""

    def __init__(self, problem, degree=1):
          super(FemBemFKSolver,self).__init__(problem, degree)
          self.doftionary = self.get_boundary_dof_coordinate_dict()

    def solve(self):
        """
        Return the magnetic scalar potential :math:`phi` of
        equation (2.13) in Andreas Knittel's thesis, 
        using a FemBem solver and the Fredkin-Koehler approach.

        """
        # Compute phi1 according to Knittel (2.28 - 2.29)
        self.compute_phi1(self.M, self.V)

        # Compute phi2 on the boundary
        self.__solve_phi2_boundary(self.doftionary)

        # Compute Laplace's equation (Knittel, 2.30)
        self.solve_laplace_inside(self.phi2)

        # Knittel (2.27), phi = phi1 + phi2
        self.phi = self.calc_phitot(self.phi1, self.phi2)
        return self.phi

    def __solve_phi2_boundary(self, doftionary):
        """Compute phi2 on the boundary."""
        B = self.__build_BEM_matrix(doftionary)
        phi1 = self.restrict_to(self.phi1.vector().array(), doftionary.keys())
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

        info_blue("Building Boundary Element Matrix")
        for i, dof in enumerate(doftionary):
            bar.update(i)
            BEM[i] = self.__get_BEM_row(doftionary[dof], keys, i)
        return BEM

    def __get_BEM_row(self, R, dofs, i):
        """Return row i in the BEM."""
        v = self.__BEMnumerator(R)
        w = self.__BEMdenominator(R)
        psi = TestFunction(self.V)

        # First term containing integration part (Knittel, eq. (2.52)) 
        L = 1.0/(4*DOLFIN_PI)*psi*v*w*ds 
        
        # Create a row from the first term
        bigrow = assemble(L, form_compiler_parameters=self.ffc_options)
        row = self.restrict_to(bigrow, dofs)
        
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
        v = Expression(v, **kwargs)

        W = VectorFunctionSpace(self.problem.mesh, "CR", self.degree, dim=dim)
        v = interpolate(v, W)
        n = FacetNormal(self.V.mesh())
        v = dot(v, n)
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
        return Expression(w, **kwargs)

    def __solid_angle(self):
        """
        TODO: Write solid angle function here.

        """
        SA = 0
 
        # Solid angle must not exceed the unit circle
        assert abs(SA) <= 2*np.pi

        return SA
          

class FKSolverTrunc(sb.TruncDeMagSolver,FKSolver):
    """FK Solver using domain truncation"""
    ###Only partially implemented at the moment
    def __init__(self,problem, degree = 1):
        self.problem = problem
        self.degree = degree
        
    def solve(self):
        #Set up spaces,functions, measures etc.
        V = FunctionSpace(self.problem.mesh,"CG",self.degree)
        if self.problem.mesh.topology().dim() == 1:
            Mspace = FunctionSpace(self.problem.mesh,"DG",self.degree)
        else:
            Mspace = VectorFunctionSpace(self.problem.mesh,"DG",self.degree)
        phi0 = Function(V)
        phi1 = Function(V)

        dxC = self.problem.dxC
        dSC = self.problem.dSC
        N = FacetNormal(self.problem.coremesh)

        #Define the magnetisation
        M = interpolate(Expression(self.problem.M),Mspace)

        ########################################
        #Solve for phi0
        ########################################
##        #A boundary point used to specify the pure neumann problem
        r = self.problem.r
        class BoundPoint(SubDomain):
            def inside(self, x, on_boundary):
                return near(x[0], 0.5 - r)

        dbc1 = DirichletBC(V, 0.0, BoundPoint())

        #Forms for Neumann Poisson Equation for phi0

        u = TrialFunction(V)
        v = TestFunction(V)
        a = dot(grad(u),grad(v))*dxC
        f = (div(M)*v)*dxC  #Source term in core
        f += (dot(M,N)*v)('-')*dSC   #Neumann Conditions on edge of core

        A = assemble(a,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)
        F = assemble(f,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)

        dbc1.apply(A,F)
        A.ident_zeros()
        print A.array()
        solve(A,phi0.vector(),F)

        ########################################
        #Solve for phi1
        ########################################
        L = FunctionSpace(self.problem.mesh,"CG",self.degree)
        VD = FunctionSpace(self.problem.mesh,"DG",self.degree)
        W = MixedFunctionSpace((V,L))
        u,l = TrialFunctions(W)
        v,q = TestFunctions(W)
        sol = Function(W)

        #Forms for phi1
        a = dot(grad(u),grad(v))*dx
        f = q('-')*phi0('-')*dSC
        a += q('-')*jump(u)*dSC #Jump in solution on core boundary
        a += (l*v)('-')*dSC

        #Dirichlet BC at our approximate boundary
        dbc = DirichletBC(W.sub(0),0.0,"on_boundary")

        A = assemble(a,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)
        F = assemble(f,cell_domains = self.problem.corefunc, interior_facet_domains = self.problem.coreboundfunc)

        dbc.apply(A)
        dbc.apply(F)
        A.ident_zeros()
        solve(A, sol.vector(),F)
        solphi,sollag = sol.split()
        phi1.assign(solphi)

        phitot = Function(V)
        print phi0.vector().array()
        print phi1.vector().array()
        phitot.vector()[:] = phi0.vector() + phi1.vector()

        #Store Variables for outside testing
        self.V = V
        self.phitot = phitot
        self.phi0 = phi0
        self.phi1 = phi1
        self.sol = sol
        self.M = M
        self.Mspace = Mspace
        return phitot

if __name__ == "__main__":
    
    from finmag.demag.problems import prob_fembem_testcases as pft
    problem = pft.MagUnitSphere()
    solver = FemBemFKSolver(problem)
    phi = solver.solve()
    plot(phi, interactive=True)
    
    Hdemag = solver.get_demagfield(phi)
    x, y, z = Hdemag.split(True)
    x, y, z = x.vector().array(), y.vector().array(), z.vector().array()
    for i in range(len(x)):
        print x[i], y[i], z[i]

    print "Max values: x:%g, y:%g, z:%g" % (max(x), max(y), max(z))
    print "Min values: x:%g, y:%g, z:%g" % (min(x), min(y), min(z))
    print "Avg values: x:%g, y:%g, z:%g" % (np.average(x), np.average(y), np.average(z))

