"""
Test the precision of the fembemsolvers using various quadrature orders. 
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
import finmag.demag.tests.analytic_solutions as asol

##########################################
#Extended solver classes
#########################################
class FemBemGCRSplitSolver(solver_gcr.FemBemGCRSolver):
    def solve(self,):
        super(FemBemGCRSplitSolver,self).solve()
        #Split the Demagfield into Component functions
        self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True)

class FemBemGCRSolver1(FemBemGCRSplitSolver):
    def __init__(self,problem):
        super(FemBemGCRSolver1,self).__init__(problem)
        self.ffc_options = {"quadrature_rule":"canonical","quadrature_degree":1}

class FemBemGCRSolver4(FemBemGCRSplitSolver):
    def __init__(self,problem):
        super(FemBemGCRSolver4,self).__init__(problem)
        self.ffc_options = {"quadrature_rule":"canonical","quadrature_degree":2}

class FemBemGCRSolver7(FemBemGCRSplitSolver):
    def __init__(self,problem):
        super(FemBemGCRSolver7,self).__init__(problem)
        self.ffc_options = {"quadrature_rule":"canonical","quadrature_degree":1}

class FemBemGCRSolver10(FemBemGCRSplitSolver):
    def __init__(self,problem):
        super(FemBemGCRSolver10,self).__init__(problem)
        self.ffc_options = {"quadrature_rule":"canonical","quadrature_degree":1} 

#This class mimics a FemBemDeMagSolver but returns analytical values.
UniformDemagSphere = asol.UniformDemagSphere
 
##########################################
#Test parameters
#########################################

#Mesh fineness, ie UnitSphere(n)
finenesslist = range(2,8)
#Create a problem for each level of fineness
problems = [pft.MagUnitSphere(n) for n in finenesslist]

#Xaxis - Number of finite elements
numelement = [p.mesh.num_cells() for p in problems]
xaxis = ("Number of elements",numelement)

#Solvers
test_solver_classes = {"QO 1": FemBemGCRSolver1,"QO 4": FemBemGCRSolver4,"QO 7": FemBemGCRSolver7,"QO 10": FemBemGCRSolver10}
reference_solver_class = {"Analytical":UniformDemagSphere}

#Test solutions
test_solutions = {"Phi":"phi","Hdemag":"Hdemag"}
#Norms
norms = {"L2 Error":en.L2_error,"Discrete Max Error":en.discrete_max_error}

#cases
cases = [("Phi","L2 Error"),("Phi","Discrete Max Error"),("Hdemag","L2 Error")]

#Create a ConvergenceTester and generate a report
ct = ct.ConvergenceTester(test_solver_classes,reference_solver_class,test_solutions,problems,norms,xaxis,cases)
ct.print_report()
ct.plot_results()
