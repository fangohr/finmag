"""Standard Demagnetisation Testproblems for truncated solvers""" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"
        
from dolfin import *
import numpy as np
import prob_base as pb

class MagUnitInterval(pb.TruncDemagProblem):
    """Create 1d test problem where define a mesh,
    and a part of the mesh has been marked to be vacuum (with 0) and
    a part has been marked to be the ferromagnetic body (with 1).

    Can later replace this with a meshfile generated with an external 
    mesher.

    Once the constructor calls the constructor of the base class (TruncDemagProblem), we also
    have marked facets.
    """
    def __init__(self,n=10):
        mesh = UnitInterval(n)
        self.r = 0.1 #Radius of magnetic Core
        self.gamma = 700 #Suggested parameter for nitsche solver
        r = self.r

        class IntervalCore(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + r + DOLFIN_EPS and x[0] > 0.5 - r - DOLFIN_EPS

        #TODO: Make M into a 3d vector here
        M = "1"

        #Initialize Base Class
        super(MagUnitInterval,self).__init__(mesh, IntervalCore(),M)

    def desc(self):
        return "unit interval demagnetisation test problem"

class MagUnitCircle(pb.TruncDemagProblem):
    def __init__(self,n=10):
        mesh = UnitCircle(n)
        self.r = 0.2 #Radius of magnetic Core
        self.gamma = 13.0 #Suggested parameter for nitsche solver
        r = self.r
        class CircleCore(SubDomain):
            def inside(self, x, on_boundary):
                return np.linalg.norm(x,2) < r + DOLFIN_EPS

        #TODO Make M three dimensional
        M = ("1","0")
        #Initialize Base Class
        super(MagUnitCircle,self).__init__(mesh,CircleCore(),M)
    def desc(self):
        return "unit circle demagnetisation test problem"

class MagUnitSphere(pb.TruncDemagProblem):
    def __init__(self,n=10):
        mesh = UnitSphere(10)
        self.r = 0.2 #Radius of magnetic Core
        self.gamma = 0.9 #Suggested parameter for nitsche solver
        r = self.r
        class SphereCore(SubDomain):
            def inside(self, x, on_boundary):
                return x[0]*x[0] + x[1]*x[1] + x[2]*x[2] < r*r + DOLFIN_EPS
        M = ("1","0","0")
        #Initialize Base Class
        super(MagUnitSphere,self).__init__(mesh,SphereCore(),M)
    def desc(self):
        return "unit sphere demagnetisation test problem"

if __name__ == "__main__":
    problem = MagUnitSphere()
    print problem.r
    print problem.coremesh.coordinates()
    plot(problem.coremesh)
    interactive()
