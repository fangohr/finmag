"""Standard Demagnetisation Testproblems for FEMBEM"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

import os
from dolfin import *
import finmag.util.meshes as meshes

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

class MagUnitCircle(object):
    def __init__(self, n= 20):
        self.mesh = UnitCircle(n)
        self.M = ("1","0") #TODO Make M three dimensional
        self.Ms = 1
    def desc(self):
        return "unit circle demagnetisation test problem fembem"
    
# XXX TODO: This should probably be merged with MagSphereBase below?!
class MagUnitSphere(object):
    """Uniformly magnetized sphere problem for fembem solvers"""
    def __init__(self, maxh=0.5):
        self.mesh = meshes.sphere(1.0, maxh=maxh, directory=MODULE_DIR)
        #M = ("1","0","0")
        self.Ms = 1
        self.maxh = maxh
        self.M = (str(self.Ms), "0", "0")
    def desc(self):
        return "Unit sphere demagnetisation test problem fembem, maxh={}, Ms={}".format(self.maxh, self.Ms)

class MagUnitIntervalMesh(object):
    """Create 1d test problem where define a mesh,
    and a part of the mesh has been marked to be vacuum (with 0) and
    a part has been marked to be the ferromagnetic body (with 1).

    Can later replace this with a meshfile generated with an external 
    mesher.

    Once the constructor calls the constructor of the base class (TruncDemagProblem), we also
    have marked facets.
    """
    def __init__(self, n=10):
        self.mesh = UnitIntervalMesh(n)

        #TODO: Make M into a 3d vector here
        self.M = "1"
        self.Ms = 1

###########################################################
#This Section contains better quality unit sphere meshes
#and problems that use them.    
###########################################################
class MagSphereBase(object):
    """Base class for MagSphere classes"""
    def __init__(self,maxh,radius=10):
        self.mesh = meshes.sphere(radius, maxh, directory=MODULE_DIR)
        self.Ms = 1.0
        self.m = (str(self.Ms), "0.0", "0.0")
        self.M = self.m # do we need this?
        self.V = VectorFunctionSpace(self.mesh, "CG", 1)
        self.m = interpolate(Expression(self.m), self.V)
        self.r = radius        
        self.maxh = maxh
        
    def desc(self):
        return "Sphere demag test problem, Ms=%g, radius=%g, maxh=%g" %(self.Ms, self.r, self.maxh)
