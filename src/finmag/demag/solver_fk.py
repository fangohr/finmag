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

class FemBemFKSolver(sb.FemBemDeMagSolver):
    """FemBem solver for Demag Problems using the Fredkin-Koehler approach.""" 

    def __init__(self, problem, degree=1):
        super(FemBemFKSolver,self).__init__(problem, degree)
        #Default linalg solver parameters
        self.phi1solverparams = {"linear_solver":"lu"}
        self.phi2solverparams = {"linear_solver":"lu"}
        self.phi1 = Function(self.V)
        self.phi2 = Function(self.V)
        self.degree = degree

    def compute_phi1(self):
        """
        Get phi1 defined over the mesh with the point given the value 0.
        phi1 is the solution to the inhomogeneous Neumann problem.
        M is the magnetisation field.
        V is a FunctionSpace

        """
        # Define functions
        n = FacetNormal(self.V.mesh())
        
        # Define forms
        eps = 1e-8
        a = dot(grad(self.u),grad(self.v))*dx - dot(eps*self.u,self.v)*dx 
        f = dot(n, self.M)*self.v*ds - div(self.M)*self.v*dx

        # Solve for the DOFs in phi1
        solve(a == f, self.phi1,solver_parameters = self.phi1solverparams)
    
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
        self.solve_laplace_inside(self.phi2,self.phi2solverparams)

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

