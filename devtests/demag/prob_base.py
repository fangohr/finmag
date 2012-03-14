"""Base classes for Demagnetisation Problems"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"
        
from dolfin import *
import numpy as np
from interiorboundary import InteriorBoundary

class DeMagProblem(object):
    """Base class for all demag problems"""
    def __init__(self,mesh,M):
        self.mesh = mesh
        self.M = M

class TruncDeMagProblem(DeMagProblem):
    """Base class for demag problems with truncated domains"""
    def __init__(self,mesh,subdomain,M):
        """
        - mesh is the problem mesh
        
        - subdomain: Subdomain an object of type SubDomain which defines an
          inside of the mesh (to mark the magnetic region)

        - and M the initial magnetisation (Expression at the moment)
        """
        
        #(Currently M is constant)
        super(TruncDeMagProblem,self).__init__(mesh,M)
        self.subdomain = subdomain
        self.calculate_subsandbounds()

    def calculate_subsandbounds(self):
        """Calulate the submeshs and their common boundary"""
        
        #Mesh Function
        self.corefunc = MeshFunction("uint", self.mesh, self.mesh.topology().dim())
        self.corefunc.set_all(0)
        self.subdomain.mark(self.corefunc,1)
        
        #generate submesh for the core and vacuum
        self.coremesh = SubMesh(self.mesh,self.corefunc,1)
        self.vacmesh = SubMesh(self.mesh,self.corefunc,0)

        #generate interior boundary
        self.corebound = InteriorBoundary(self.mesh)
        self.corebound.create_boundary(self.coremesh)
        self.coreboundfunc = self.corebound.boundaries[0]

        #Store Value of coreboundary number as a constant
        self.COREBOUNDNUM = 2

        #generate measures
        self.dxC = dx(1)  #Core
        self.dxV = dx(0)  #Vacuum
        self.dSC = dS(self.COREBOUNDNUM)  #Core Boundary

    def refine_core(self):
        """Refine the Mesh inside the Magnetic Core"""
        #Mark the cells in the core
        cell_markers = CellFunction("bool", self.mesh)
        cell_markers.set_all(False)
        self.subdomain.mark(cell_markers,True)
        #Refine
        self.mesh = refine(self.mesh, cell_markers)
        #Regenerate Subdomains and boundaries
        self.calculate_subsandbounds()
        
    def create_fembem_problem(self):
        """Generate a FEMBEM problem over the magnetic core submesh"""
        return FemBemDeMagProblem(self.coremesh,self.M)

class FemBemDeMagProblem(DeMagProblem):
    """Base class for FEMBEM demag problems"""
    def __init__(self,mesh,M):
        super(FemBemDeMagProblem,self).__init__(mesh,M)
    
    
