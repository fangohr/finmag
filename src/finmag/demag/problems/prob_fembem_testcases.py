"""Standard Demagnetisation Testproblems for FEMBEM"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np
import finmag.demag.problems.prob_base as pb
import finmag.util.convert_mesh as cm
import os

#TODO need a more exciting M, GCR solver has phiA = 0 due to
#divM = 0 if M constant
class MagUnitCircle(pb.FemBemDeMagProblem):
    def __init__(self,n= 20):
        mesh = UnitCircle(n)
        #TODO Make M three dimensional
        M = ("1","0")
        #Initialize Base Class
        super(MagUnitCircle,self).__init__(mesh,M)
    def desc(self):
        return "unit circle demagnetisation test problem fembem"
    
class MagUnitSphere(pb.FemBemDeMagProblem):
    """Uniformly magnetized sphere problem for fembem solvers"""
    def __init__(self, n=10):
        mesh = UnitSphere(n)
        #M = ("1","0","0")
        self.Ms = 1
        self.n = n
        M = (str(self.Ms), "0", "0")
        #Initialize Base Class
        super(MagUnitSphere,self).__init__(mesh,M)
        
    def desc(self):
        return "Unit sphere demagnetisation test problem fembem, n=%d, Ms=%g" % (self.n, self.Ms)

class MagUnitInterval(pb.FemBemDeMagProblem):
    """Create 1d test problem where define a mesh,
    and a part of the mesh has been marked to be vacuum (with 0) and
    a part has been marked to be the ferromagnetic body (with 1).

    Can later replace this with a meshfile generated with an external 
    mesher.

    Once the constructor calls the constructor of the base class (TruncDemagProblem), we also
    have marked facets.
    """
    def __init__(self, n=10):
        mesh = UnitInterval(n)

        #TODO: Make M into a 3d vector here
        M = "1"

        #Initialize Base Class
        super(MagUnitInterval,self).__init__(mesh,M)

###########################################################
#This Section contains better quality unit sphere meshes
#and problems that use them. Unfortunatly, it can take
#a while to generate one of these meshes for the first time.     
###########################################################

class MagSphereBase(pb.FemBemDeMagProblem,cm.GeoMeshSphereProblem):
    """Base class for MagSphere classes"""
    def __init__(self,maxh):
        #Try to regenerate meshes if need be
        cm.GeoMeshSphereProblem.__init__(self,maxh)
        
        mesh = Mesh("../../mesh/sphere"+str(maxh)+".xml.gz")
        self.Ms = 1.0
        M = (str(self.Ms), "0.0", "0.0")

        #Initialize Base pb.FemBemDeMagProblem
        pb.FemBemDeMagProblem.__init__(self,mesh,M)
        
    def desc(self):
        return "Sphere demagnetisation test problem fembem, Ms=%g, maxh = %g" %(self.Ms,self.maxh)
    
#Note python doesn't allow :." in class names so the Sphere1.0 is now Sphere10 etc...        
class MagSphere(MagSphereBase):
    """Demag Sphere problem using the sphere mesh from nmag example. maxh = 1.0"""
    def __init__(self):
        MagSphereBase.__init__(self,1.0)

class MagSphere50(MagSphereBase):
    """Demag Sphere problem Using the geo sphere mesh maxh  = 5.0"""
    def __init__(self):
        MagSphereBase.__init__(self,5.0)

class MagSphere25(MagSphereBase):
    """Demag Sphere problem Using the geo sphere mesh maxh  = 2.5"""
    def __init__(self):
        MagSphereBase.__init__(self,2.5)

class MagSphere20(MagSphereBase):
    """Demag Sphere problem Using the geo sphere mesh maxh  = 2.0"""
    def __init__(self):
        MagSphereBase.__init__(self,2.0)
S = MagSphere20()    

