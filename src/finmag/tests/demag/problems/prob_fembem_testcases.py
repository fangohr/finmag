"""Standard Demagnetisation Testproblems for FEMBEM"""

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__project__ = "Finmag"
__organisation__ = "University of Southampton"

from dolfin import *
from finmag.util.convert_mesh import convert_mesh
import prob_base as pb
import os
import finmag.mesh.marker as mark

#TODO need a more exciting M, GCR solver has phiA = 0 due to
#divM = 0 if M constant
class MagUnitCircle(object):
    def __init__(self, n= 20):
        self.mesh = UnitCircle(n)
        self.M = ("1","0") #TODO Make M three dimensional
        self.Ms = 1
    def desc(self):
        return "unit circle demagnetisation test problem fembem"
    
class MagUnitSphere(object):
    """Uniformly magnetized sphere problem for fembem solvers"""
    def __init__(self, n=10):
        self.mesh = UnitSphere(n)
        #M = ("1","0","0")
        self.Ms = 1
        self.n = n
        self.M = (str(self.Ms), "0", "0")
    def desc(self):
        return "Unit sphere demagnetisation test problem fembem, n=%d, Ms=%g" % (self.n, self.Ms)

class MagUnitInterval(object):
    """Create 1d test problem where define a mesh,
    and a part of the mesh has been marked to be vacuum (with 0) and
    a part has been marked to be the ferromagnetic body (with 1).

    Can later replace this with a meshfile generated with an external 
    mesher.

    Once the constructor calls the constructor of the base class (TruncDemagProblem), we also
    have marked facets.
    """
    def __init__(self, n=10):
        self.mesh = UnitInterval(n)

        #TODO: Make M into a 3d vector here
        self.M = "1"
        self.Ms = 1

###########################################################
#This Section contains better quality unit sphere meshes
#and problems that use them.    
###########################################################

class MagSphereBase(object):
    def generate_mesh(self,pathmesh,geofile):
        """
        Checkes the path pathmesh to see if the file exists,
        if not it is generated in the path pathmesh using
        the information from the geofile.
        """
        #Check if the meshpath is of type .gz
        name, type_ = os.path.splitext(pathmesh)
        if type_ != '.gz':
            print 'Only .gz files are supported as input by the class MeshGenerator.\
                    Feel free to rewrite the class and make it more general'
            sys.exit(1)
            
        #Generate the mesh if it does not exist    
        if not os.path.isfile(pathmesh):
            #Remove the .xml.gz ending
            pathgeo = pathmesh.rstrip('.xml.gz')
            #Add the ".geo"
            pathgeo = "".join([pathgeo,".geo"])
            #Create a geofile
            f = open(pathgeo,"w")
            f.write(geofile)                   
            f.close()

            #call the mesh generation function
            convert_mesh(pathgeo)
            #the file should now be in 
            #pathmesh#

            #Delete the geofile file
            print "removing geo file"
            os.remove(pathgeo)

    """Base class for MagSphere classes"""
    def __init__(self,maxh,radius=10):
        #Try to regenerate mesh files if need be
        geofile = "algebraic3d \n \n \
                   solid main = sphere (0, 0, 0; "+str(radius)+\
                   ")-maxh="+str(maxh)+" ; \n \n \
                   tlo main;"

        #Get rid of "." in the file name as this confuses other programs
        maxhstr = str(maxh).replace(".","dot")
        radiusstr=str(radius).replace(".","dot")
        
        meshpath = "".join([os.path.dirname(mark.__file__),"/","sphere-",maxhstr,"-",
                            radiusstr,\
                            ".xml.gz"])



        self.generate_mesh(meshpath,geofile)
        #Upload the dolfin mesh
        self.mesh = Mesh(meshpath)
        self.Ms = 1.0
        self.m = (str(self.Ms), "0.0", "0.0")
        self.M = self.m # do we need this?
        self.V = VectorFunctionSpace(self.mesh, "CG", 1)
        self.m = interpolate(Expression(self.m), self.V)
        self.r = radius        
        self.maxh = maxh

        
    def desc(self):
        return "Sphere demag test problem, Ms=%g, radius=%g, maxh=%g" %(self.Ms, self.r, self.maxh)
