"""
Test the precision of the fembemsolvers using various quadrature orders. 
"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from finmag.energies.demag import solver_gcr
from finmag.tests.demag.problems import prob_fembem_testcases as pft
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

class FemBemGCRSolver2(FemBemGCRSplitSolver):
    def __init__(self,problem):
        super(FemBemGCRSolver2,self).__init__(problem)
        self.ffc_options = {"quadrature_rule":"canonical","quadrature_degree":2}

class FemBemGCRSolver4(FemBemGCRSplitSolver):
    def __init__(self,problem):
        super(FemBemGCRSolver4,self).__init__(problem)
        self.ffc_options = {"quadrature_rule":"canonical","quadrature_degree":4}

class FemBemGCRSolver6(FemBemGCRSplitSolver):
    def __init__(self,problem):
        super(FemBemGCRSolver6,self).__init__(problem)
        self.ffc_options = {"quadrature_rule":"canonical","quadrature_degree":6} 

#This class mimics a FemBemDeMagSolver but returns analytical values.
UniformDemagSphere = asol.UniformDemagSphere
 
##########################################
#Test parameters
#########################################
#High Quality geo meshes

sphere = lambda hmax: MagSphereBase(hmax, r=10)
problems = [sphere(5.0), sphere(2.5), sphere(2.0), sphere(1.5), sphere(1.2), sphere(1.0)]

#Xaxis - Number of finite elements
numvertex = [p.mesh.num_vertices() for p in problems]
xaxis = ("Number of elements",numvertex)

#Solvers
test_solver_classes = {"QO 1": FemBemGCRSolver1,"QO 2": FemBemGCRSolver2,\
                       "QO 4": FemBemGCRSolver4,"QO 6": FemBemGCRSolver6}
reference_solver_class = {"Analytical":UniformDemagSphere}

#Test solutions
test_solutions = {"Phi":"phi","Hdemag":"Hdemag","Hdemag X":"Hdemagx"}
#Norms
norms = {"L2 Error":en.L2_error,"Discrete Max Error":en.discrete_max_error}

#cases
cases = [("Phi","L2 Error"),("Phi","Discrete Max Error"),("Hdemag","L2 Error"),("Hdemag X","Discrete Max Error")]

#Create a ConvergenceTester and generate a report
ct = ct.ConvergenceTester(test_solver_classes,reference_solver_class,test_solutions,problems,norms,xaxis,cases,subplots = "22" )
ct.print_report()
ct.plot_results()
