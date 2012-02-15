#A Unit Interval with Uniform Magnetisation

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__license__  = "GNU GPL Version 3 or any later version"
        
from dolfin import *
from prob_demag import *
import numpy as np
class MagUnitInterval(TruncDemagProblem):
    def __init__(self):
        mesh = UnitInterval(10)
        self.r = 0.1 #Radius of magnetic Core
        r = self.r
        class Sphere1d(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + r + DOLFIN_EPS and x[0] > 0.5 - r - DOLFIN_EPS
        M = "1"
        #Initialize Base Class
        TruncDemagProblem.__init__(self,mesh, Sphere1d(),M)

class MagUnitCircle(TruncDemagProblem):
    def __init__(self):
        mesh = UnitCircle(20)
        self.r = 0.2 #Radius of magnetic Core
        r = self.r
        class Sphere2d(SubDomain):
            def inside(self, x, on_boundary):
                return np.linalg.norm(x,2) < r
        M = ("1","0")
        #Initialize Base Class
        TruncDemagProblem.__init__(self,mesh,Sphere2d(),M)

##problem = MagUnitCircle()
##print problem.r
