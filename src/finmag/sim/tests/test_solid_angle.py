import numpy as np
from finmag.util.solid_angle_magpar import return_csa_magpar
from finmag.native import llg as native_llg

TOLERANCE = 1e-15

#this is the function to compute the solid angle:
csa = native_llg.compute_solid_angle
csa_magpar = return_csa_magpar()

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
    print "T",T.shape,"\n",T
    return csa(r,T)

#print "Random tests"
#print solid_angle([[1.,0,0],[-0.5,-0.5,0],[-0.5,0.5,0]],[0,0,-100])
#print solid_angle([[1.,0,0],[-0.5,-0.5,0],[-0.5,0.5,0]],[0,0,-1e-1])
#print solid_angle([[1.,0,0],[-0.5,-0.5,0],[-0.5,0.5,0]],[1,0,0])

#print "the following triangle is big, so it should cover 'half the sky', i.e. 2*pi=",math.pi*2
#a=1000.
#print solid_angle([[a,-a,-a],[0,-a,a],[0,0,0]],[0,0,1e-5])
#print "looking from the other side should give us the solid same angle?"
#print solid_angle([[a,-a,-a],[0,-a,a],[0,0,0]],[0,0,-1e-5])

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
    print csa(r,T)

def test_solid_angle_first_octant():
    """
    In spherical coordinates, the solid angle is defined as
    :math:`\\d\\Omega=\\sin\\theta\\,d\\theta\\,d\\varphi`.

    From the point (0, 0, 0), the triangle defined by the vertices
    (1, 0, 0), (0, 1, 0) and (0, 0, 1) in cartesian coordinates
    can be identified with the azimuth and zenith angles PI/2
    and the integral yields PI/2.

    """
    origin = np.zeros((3, 1))

    # first octant (+, +, +) front-right-top
    triangle = np.array([[[1.],[0.],[0.]], [[0.],[1.],[0.]], [[0.],[0.],[1.]]])
    assert triangle.shape == (3, 3, 1)
    angle = csa(origin, triangle)
    assert abs(angle[0] - np.pi / 2) < TOLERANCE, \
            "The solid angle is {0}, but should be PI/2={1}.".format(angle[0], np.pi/2)

    magpar = csa_magpar(np.zeros(3), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    assert abs(angle[0] - magpar) < TOLERANCE

def test_solid_angle_one_minus_sign():
    origin = np.zeros((3, 1))

    # (-, +, +) back-right-top
    triangle = np.array([[[-1.],[0.],[0.]], [[0.],[1.],[0.]], [[0.],[0.],[1.]]])
    assert triangle.shape == (3, 3, 1)
    angle = csa(origin, triangle)[0]
    magpar = csa_magpar(np.zeros(3), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    assert abs(angle - np.pi / 2) < TOLERANCE, \
            "The solid angle is {0}, but should be PI/2={1}.".format(angle[0], np.pi/2)

    assert abs(angle - magpar) < TOLERANCE

def test_solid_angle_two_minus_signs():
    origin = np.zeros((3, 1))

    # (-, -, +) back-left-top
    triangle = np.array([[[-1.],[0.],[0.]], [[0.],[-1.],[0.]], [[0.],[0.],[1.]]])
    assert triangle.shape == (3, 3, 1)
    angle = csa(origin, triangle)
    assert abs(angle[0] - np.pi / 2) < TOLERANCE, \
            "The solid angle is {0}, but should be PI/2={1}.".format(angle[0], np.pi/2)

    magpar = csa_magpar(np.zeros(3), np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, 1]))
    assert abs(angle[0] - magpar) < TOLERANCE

def test_octants_solid_angle():
    """
    By the same reasing as above, we get 4PI for the solid angle of a sphere
    as seen by one point inside of it.

    That means that another way of getting the solid angle of a single octant
    is considering we are looking at one of the 8 divisions of the
    euclidean three-dimensional coordinate system, divide the solid
    angle of a sphere by 8 and get PI/2.

    """
    origin = np.zeros((3, 1))

    TRIANGLES = 8
    triangles = np.zeros((3, 3, TRIANGLES))
    # (3, 3, 8) three components, three vertices per triangle, 8 triangles

    triangles[0,0,] = np.array([1., -1., 1., -1., 1., -1., 1., -1.]) # X, V1, 1-8
    triangles[1,1,] = np.array([1., 1., -1., -1., 1., 1., -1., -1.]) # Y, V2, 1-8
    triangles[2,2,] = np.array([1., 1., 1., 1., -1., -1., -1., -1.]) # Z, V3, 1-8

    angle = csa(origin, triangles)
    assert abs(angle[0] - 4*np.pi) < TOLERANCE, \
        "The solid angle is {0}, but should be 4PI={1}.".format(angle[0], 4*np.pi)

    front  = np.array([ 1.0,  0.0,  0.0])
    back   = np.array([-1.0,  0.0,  0.0])
    left   = np.array([ 0.0, -1.0,  0.0])
    right  = np.array([ 0.0,  1.0,  0.0])
    top    = np.array([ 0.0,  0.0,  1.0])
    bottom = np.array([ 0.0,  0.0, -1.0])
    
    magpar = 0.0
    for abscissa in [front, back]:
        for ordinate in [left, right]:
            for applicate in [top, bottom]:
                magpar += csa_magpar(np.zeros(3), abscissa, ordinate, applicate)
    assert abs(angle[0] - magpar) < TOLERANCE

if __name__=="__main__":
    for name in globals().keys():
#        if name.startswith("test_solid_angle_invariance"):
        if name.startswith("test_"):
            print "Running", name
            globals()[name]()
