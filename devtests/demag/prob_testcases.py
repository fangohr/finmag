#Standard Demagnetisation Testproblems 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"
        
from dolfin import *
from prob_demag import *
import numpy as np
class MagUnitInterval(TruncDemagProblem):
    def __init__(self):
        mesh = UnitInterval(10)
        self.r = 0.1 #Radius of magnetic Core
        self.gamma = 700 #Suggested parameter for nitsche solver
        r = self.r
        class IntervalCore(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + r + DOLFIN_EPS and x[0] > 0.5 - r - DOLFIN_EPS
        M = "1"
        #Initialize Base Class
        TruncDemagProblem.__init__(self,mesh, IntervalCore(),M)
    def desc(self):
        return "unit interval demagnetisation test problem"

class MagUnitCircle(TruncDemagProblem):
    def __init__(self):
        mesh = UnitCircle(10)
        self.r = 0.2 #Radius of magnetic Core
        self.gamma = 13.0 #Suggested parameter for nitsche solver
        r = self.r
        class MagUnitCircle(SubDomain):
            def inside(self, x, on_boundary):
                return np.linalg.norm(x,2) < r + DOLFIN_EPS
        M = ("1","0")
        #Initialize Base Class
        TruncDemagProblem.__init__(self,mesh,MagUnitCircle(),M)
    def desc(self):
        return "unit circle demagnetisation test problem"

class MagUnitSphere(TruncDemagProblem):
    def __init__(self):
        mesh = UnitSphere(10)
        self.r = 0.2 #Radius of magnetic Core
        self.gamma = 0.9 #Suggested parameter for nitsche solver
        r = self.r
        class SphereCore(SubDomain):
            def inside(self, x, on_boundary):
                return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] < r*r + DOLFIN_EPS
        M = ("1","0","0")
        #Initialize Base Class
        TruncDemagProblem.__init__(self,mesh,SphereCore(),M)
    def desc(self):
        return "unit sphere demagnetisation test problem"

if __name__ == "__main__":
    problem = MagUnitSphere()
    print problem.r
    print problem.coremesh.coordinates()
    plot(problem.coremesh)
    interactive()
