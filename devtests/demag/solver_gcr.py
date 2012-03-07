"""Solvers for the demagnetization field using the Garcia-Cervera-Roma approach""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import solver_base as sb

class GCRDeMagSolver(sb.DeMagSolver):
     """Class containing methods shared by GCR solvers"""
     def __init__(self,problem,degree = 1):
         super(GCRDeMagSolver,self).__init__(problem,degree)
         #Define the two potentials
         self.phia = Function(self.V)
         self.phib = Function(self.V)
         
class GCRFemBemDeMagSolver(GCRDeMagSolver,sb.FemBemDeMagSolver):
    """FemBem solver for Demag Problems using the GCR approach"""
    
    def __init__(self,problem,degree = 1):
        super(GCRFemBemDeMagSolver,self).__init__(problem,degree)
    
    def solve_phia(self,method = "lu"):
        """Solve for potential phia in the magentic region using FEM"""
        #Define functions
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        #Define forms
        a = dot(grad(u),grad(v))*dx
        f = (-div(M)*v)*dx  #Source term

        #Define Boundary Conditions
        bc = DirichelBC(self.V,0,"on_boundary")

        #solve for phia
        A,F = assemble(a,f)

        solve(A,self.phia.vector(),F,method)

if __name__ == "__main__":
    import prob_fembem_testcases as pft
    problem = pft.MagUnitSphere()
    solver = GCRFemBemDeMagSolver(problem)
    solver.solve_phia
    plot(solver.phia, title = "PhiA from GCR Demag method")
    interactive()
