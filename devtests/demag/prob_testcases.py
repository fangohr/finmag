#Standard Demagnetisation Testproblems 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"
        
from dolfin import *
import numpy as np
from interiorboundary import InteriorBoundary

class TruncDemagProblem(object):
    
    def __init__(self,mesh,subdomain,M):
        """
        - mesh is the problem mesh
        
        - subdomain: Subdomain an object of type SubDomain which defines an
          inside of the mesh (to mark the magnetic region)

        - and M the initial magnetisation (Expression at the moment)

        """
        #(Currently M is constant)
        self.mesh = mesh
        self.subdomain = subdomain
        self.M = M

        #Mesh Function
        self.corefunc = MeshFunction("uint", mesh, mesh.topology().dim())
        self.corefunc.set_all(0)
        subdomain.mark(self.corefunc,1)
        
        #generate submesh for the core and vacuum
        self.coremesh = SubMesh(mesh,self.corefunc,1)
        self.vacmesh = SubMesh(mesh,self.corefunc,0)

        #generate interior boundary
        self.corebound = InteriorBoundary(mesh)
        self.corebound.create_boundary(self.coremesh)
        self.coreboundfunc = self.corebound.boundaries[0]
        
        #Store Value of coreboundary number as a constant
        self.COREBOUNDNUM = 2

        #generate measures
        self.dxC = dx(1)  #Core
        self.dxV = dx(0)  #Vacuum
        self.dSC = dS(self.COREBOUNDNUM)  #Core Boundary

class MagUnitInterval(TruncDemagProblem):
    """Create 1d test problem where define a mesh,
    and a part of the mesh has been marked to be vacuum (with 0) and
    a part has been marked to be the ferromagnetic body (with 1).

    Can later replace this with a meshfile generated with an external 
    mesher.

    Once the contstructor calls the contstructor of the base class (TruncDemagProblem), we also
    have marked facets.
    """
    def __init__(self):
        mesh = UnitInterval(10)
        self.r = 0.1 #Radius of magnetic Core
        self.gamma = 700 #Suggested parameter for nitsche solver
        r = self.r

        class IntervalCore(SubDomain):
            def inside(self, x, on_boundary):
                return x[0] < 0.5 + r + DOLFIN_EPS and x[0] > 0.5 - r - DOLFIN_EPS

        #TODO: Make M into a 3d vector here
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

        #TODO Make M three dimensional
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
