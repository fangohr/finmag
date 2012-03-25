"""
A Module containing analytic solution solver classes. The solvers should contain an analytic solution
to a specific type of problem, and can be used to test the accuracy of a numerical solver.
"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import finmag.demag.solver_base as sb

class UniformDemagSphere(sb.FemBemDeMagSolver):
    """
    Class containing information regarding the 3d analytical solution of a Demag Field in a uniformly
    demagnetized unit sphere with M = (1,0,0)
    """
    def __init__(self,problem):
        super(UniformDemagSphere,self).__init__(problem)
        
    def solve(self):
        self.phi = project(Expression("-x[0]/3.0"),self.V)
        self.get_demagfield()
        #Split the Demagfield into component functions
        self.Hdemagx,self.Hdemagy,self.Hdemagz = self.Hdemag.split(True) 
    
    def get_demagfield(self):
        self.Hdemag = project(Expression(("1.0/3.0","0.0","0.0")),self.Hdemagspace)
