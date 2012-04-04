"""
Test the convergence of the fembem solvers with a uniformly magnetized unit
sphere. At the moment high run time is expected so this is not yet a pytest. 

If you want to test a new solver (of class FemBemDeMagSolver) just add
the solver class to the dictionary "solvers"
"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from finmag.demag import solver_fk, solver_gcr
from finmag.demag.problems import prob_fembem_testcases as pft
import finmag.demag.tests.analytic_solutions as asol
import finmag.demag.solver_base as sb
import finmag.util.error_norms as en
import finmag.util.convergence_tester as ct

##########################################
#Extended solver classes
#########################################
#This class mimics a FemBemDeMagSolver but returns analytical values.
UniformDemagSphere = asol.UniformDemagSphere

#Extended versions of the solvers that give us some extra functions
class FemBemFKSolverTest(solver_fk.FemBemFKSolver):
    """Extended verions of FemBemFKSolver used for testing in 3d"""
    def solve(self):
        super(FemBemFKSolverTest,self).solve()
        #Split the demagfield into component functions
        self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 

class FemBemGCRSolverTest(solver_gcr.FemBemGCRSolver):
    """Extended verions of  FemBemGCRSolver used for testing in 3d"""
    def solve(self):
        super(FemBemGCRSolverTest,self).solve()
        #Split the demagfield into component functions
        self.Hdemag = self.get_demagfield(self.phi)
        self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 

##########################################
#Test parameters
#########################################

#Dolfin Meshes
#Mesh fineness, ie UnitSphere(n)
#finenesslist = [2] #range(2,5)
#Create a problem for each level of fineness
#problems = [pft.MagUnitSphere(n) for n in finenesslist]

#High Quality geo meshes
problems = [pft.MagSphere50(),pft.MagSphere25(),pft.MagSphere20(), \
            pft.MagSphere15()]#,pft.MagSphere12(),pft.MagSphere10()]

#Xaxis - Number of verticies
numvert = [p.mesh.num_vertices() for p in problems]
xaxis = ("Number of verticies",numvert)

#Solvers
test_solver_classes = {"GCR Solver": FemBemGCRSolverTest}
reference_solver_class = {"Analytical":UniformDemagSphere}

#Test solutions
test_solutions = {"Phi":"phi","Hdemag":"Hdemag","Hdemag X":"Hdemagx",\
                 "Hdemag Y":"Hdemagy","Hdemag Z":"Hdemagz"}
#Norms
norms = {"L2 Error":en.L2_error,"Discrete Max Error":en.discrete_max_error}

#cases
cases = [("Phi","L2 Error"),("Phi","Discrete Max Error"),("Hdemag","L2 Error"), \
         ("Hdemag X","Discrete Max Error"),("Hdemag Y","Discrete Max Error"),\
         ("Hdemag Z","Discrete Max Error")]

#Create a ConvergenceTester and generate a report
ct = ct.ConvergenceTester(test_solver_classes,reference_solver_class,test_solutions,problems,norms,xaxis,cases)
ct.print_report()
ct.plot_results()
