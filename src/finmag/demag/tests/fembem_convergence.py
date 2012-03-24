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
import finmag.demag.solver_base as sb
import finmag.util.error_norms as en
import finmag.util.convergence_tester as ct

##########################################
#Extended solver classes
#########################################
#This class mimics a FemBemDeMagSolver but returns analytical values.
class FemBemAnalytical(sb.FemBemDeMagSolver):
    """
    Class containing information regarding the 3d analytical solution of a Demag Field in a uniformly
    demagnetized unit sphere with M = (1,0,0)
    """
    def __init__(self,problem):
        super(FemBemAnalytical,self).__init__(problem)
        
    def solve(self):
        self.phi = project(Expression("-x[0]/3.0"),self.V)
        self.get_demagfield()
        #Split the Demagfield into Component functions
        self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 
    
    def get_demagfield(self):
        self.Hdemag = project(Expression(("1.0/3.0","0.0","0.0")),self.Hdemagspace)

#Extended versions of the solvers that give us some extra functions
class FemBemFKSolverTest(solver_fk.FemBemFKSolver):
    """Extended verions of FemBemFKSolver used for testing in 3d"""
    def solve(self):
        super(FemBemFKSolverTest,self).solve()
        #Split the Demagfield into Component functions
        self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 

class  FemBemGCRSolverTest(solver_gcr.FemBemGCRSolver):
    """Extended verions of  FemBemGCRSolver used for testing in 3d"""
    def solve(self):
        super(FemBemGCRSolverTest,self).solve()
        #Split the Demagfield into Component functions
        self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 

##########################################
#Test parameters
#########################################

#Mesh fineness, ie UnitSphere(n)
finenesslist = range(2,4)
#Create a problem for each level of fineness
problems = [pft.MagUnitSphere(n) for n in finenesslist]

#Xaxis - Number of finite elements
numelement = [p.mesh.num_cells() for p in problems]
xaxis = ("Number of elements",numelement)

#Solvers
test_solver_classes = {"FK Solver": FemBemFKSolverTest,"GCR Solver": FemBemGCRSolverTest}
reference_solver_class = {"Analytical":FemBemAnalytical}

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
