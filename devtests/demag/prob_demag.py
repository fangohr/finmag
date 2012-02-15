#The Base Class for demagnetisation problems with a truncated
#exterior domain. It is assumed there is a core domain over which
#the field M is defined, and a vacuum domain which is truncated. 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__license__  = "GNU GPL Version 3 or any later version"
        
from dolfin import *
from interiorboundary import *

class TruncDemagProblem(object):
    def __init__(self,mesh,subdomain,M):
        #Mesh is the problem mesh
        #Subdomain an object of type SubDomain which defines an
        #inside of the mesh, and M the initial magnetisation.
        #(Currently M is constant)
        self.mesh = mesh
        self.subdomain = subdomain
        self.M = M

        #Mesh Function
        corefunc = MeshFunction("uint", mesh, mesh.topology().dim())
        corefunc.set_all(0)
        subdomain.mark(corefunc,1)
        
        #generate submesh for the core and vacuum
        self.coremesh = SubMesh(mesh,corefunc,1)
        self.vacmesh = SubMesh(mesh,corefunc,0)

        #generate interior boundary
        self.corebound = InteriorBoundary(mesh)
        self.corebound.create_boundary(self.coremesh)
        self.coreboundfunc = self.corebound.boundaries[0]

        #generate measures
        self.dxC = dx(1)  #Core
        self.dxV = dx(0)  #Vacuum
        self.dSC = dS(2)  #Core Boundary
