"""Standard Demagnetisation Testproblems for FEMBEM"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
import numpy as np
import finmag.demag.problems.prob_base as pb

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

#Could also be Sphere 1.0
class MagSphere(pb.FemBemDeMagProblem):
    """Using the sphere mesh from nmag example."""
    def __init__(self):
        mesh = Mesh("../mesh/sphere10.xml")
        self.Ms = 1.0
        M = (str(self.Ms), "0.0", "0.0")

        #Initialize Base Class
        super(MagSphere,self).__init__(mesh,M)
        
    def desc(self):
        return "Sphere demagnetisation test problem fembem, Ms=%g" % self.Ms

def generate_meshes():
    """
    Creates.geo meshes in the shapes of unit spheres with
    varying mesh fineness.
    """
    #Specify meshes of varying fineness
    maxhlist = [5.0,2.5]

    #Generate missing meshes
    for maxh in maxhlist:
        #Check if the .gz file already exists
        pathgz =  "sphere" + str(maxh) + ".xml.gz"
        if not os.path.isfile(pathgz):
            pathgeo = "tempsphere" + str(maxh) + ".geo"
            #Create a geofile
            #f = open(pathgeo,"w")
            f = TemporaryFile(pathgeo,"g")
            content = "algebraic3d \n \n \
                       solid main = sphere (0, 0, 0; 10)-maxh=1.0 ; \n \n \
                       tlo main;"
            f.write(content)                   
            f.close()

            #call the mesh generation function
            convert_mesh(pathgeo)
            #the file should now be in 
            #pathgeo - "geo" + ".xml.gz"#

#As a defualt always try to regenerate missing meshes upon import
generate_meshes()
s = MagSphere()
class MagSphere50(pb.FemBemDeMagProblem):
    """Using the geo sphere mesh hmax  = 5.0"""
    def __init__(self):
        mesh = Mesh("../mesh/sphere5.0.xml")
        self.Ms = 1.0
        M = (str(self.Ms), "0.0", "0.0")

        #Initialize Base Class
        super(MagSphere,self).__init__(mesh,M)
        
    def desc(self):
        return "Sphere demagnetisation test problem fembem, Ms=%g"% self.Ms

class MagSphere25(pb.FemBemDeMagProblem):
    """Using the geo sphere mesh hmax  = 2.5"""
    def __init__(self):
        mesh = Mesh("../mesh/sphere5.0.xml")
        self.Ms = 1.0
        M = (str(self.Ms), "0.0", "0.0")

        #Initialize Base Class
        super(MagSphere,self).__init__(mesh,M)
        
    def desc(self):
        return "Sphere demagnetisation test problem fembem, Ms=%g" % self.Ms
