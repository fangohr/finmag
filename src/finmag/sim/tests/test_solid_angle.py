import math
from finmag.native import llg as native_llg


#this is the function to compute the solid angle:
csa = native_llg.compute_solid_angle

#doc string reads (in llg_module_impl.h):
csa.__doc__ = \
  """ compute_solid_angle ( r, T, a )

      Computes the solid angle subtended by the triangular mesh Ts, as seen from xs
          r - 3 x m array of points in space
          T - 3 x 3 x n array of triangular coordinates, 
              first index is for the spatial coordinate, 
              second for node number 

     a (output) -  m vector of computed solid angles

     doc string taken from llg_module_impl.h.
"""

#print help(csa)
import numpy as np

def solid_angle(triangles, point):
    """expect triangle as list of lists, if converted to numpy array should have shape 3x3
    point is a 3d- vector (the point of view for the solid angle)
    returns solid angle.
    """
    #number of solid angles to compute
    m = 1
    #number of triangles
    n = 1

    r = np.zeros((3,m))
    r[:,0] = np.array([point])
    #print "r",r.shape,
    T = np.zeros((3,3,n))
    T[:,:,0] = np.array(triangles)
    #print "T",T.shape,T
    a = np.zeros(1)
    csa(r,T,a)
    return a

#print "Random tests"
#print solid_angle([[1.,0,0],[-0.5,-0.5,0],[-0.5,0.5,0]],[0,0,-100])
#print solid_angle([[1.,0,0],[-0.5,-0.5,0],[-0.5,0.5,0]],[0,0,-1e-1])
#print solid_angle([[1.,0,0],[-0.5,-0.5,0],[-0.5,0.5,0]],[1,0,0])

print "the following triangle is big, so it should cover 'half the sky', i.e. 2*pi=",math.pi*2
a=1000.
print solid_angle([[a,0,0],[-a,-a,0],[-a,a,0]],[0,0,1e-5])
print "looking from the other side should give us the solid same angle?"
print solid_angle([[a,0,0],[-a,-a,0],[-a,a,0]],[0,0,-1e-5])

def test_solid_angle():

    #number of solid angles to compute
    m = 2
    #number of triangles
    n = 1
    r = np.zeros((3,m))
    r[:,0] = np.array([[0.,0,-1]])
    r[:,1] = np.array([[0.,0,-1e-5]])
    T = np.zeros((3,3,n))
    #T[:,:,0] = np.array([[1.,0,0],[0,-0.5,0],[0,0.5,0]])
    T[:,:,0] = np.array([[1.,0,0],[-0.5,-0.5,0],[-0.5,0.5,0]])
    print "rshape:",r.shape
    print "Tshape:",T.shape
    #number of solid_angles_to_compute
    tmp=r.shape
    if len(tmp) == 1:
        n = 1
    else:
        n = tmp[1]
    a = np.zeros(n)
    print a
    csa(r,T,a)
    print a

    
if __name__=="__main__":
    #test_solid_angle()
    pass

