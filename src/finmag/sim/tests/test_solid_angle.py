import numpy as np
from finmag.util.solid_angle_magpar import return_csa_magpar
from finmag.native import llg as native_llg

# native_llg.compute_solid_angle returns a signed angle, magpar does not.

TOLERANCE = 1e-15

csa = native_llg.compute_solid_angle
csa_magpar = return_csa_magpar()

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
    triangle = np.array([[[2.],[0.],[0.]], [[0.],[1.],[0.]], [[0.],[0.],[1.]]])
    assert triangle.shape == (3, 3, 1)
    angle = csa(origin, triangle)
    assert abs(angle[0] - np.pi / 2) < TOLERANCE, \
            "The solid angle is {}, but should be PI/2={}.".format(angle[0], np.pi/2)

    magpar = csa_magpar(np.zeros(3), np.array([2, 0, 0]), np.array([0, 5, 0]), np.array([0, 0, 1]))
    assert abs(angle - magpar) < TOLERANCE

def test_solid_angle_one_minus_sign():
    origin = np.zeros((3, 1))

    # (-, +, +) back-right-top
    triangle = np.array([[[-1.],[0.],[0.]], [[0.],[1.],[0.]], [[0.],[0.],[1.]]])
    assert triangle.shape == (3, 3, 1)
    angle = csa(origin, triangle)[0]
    magpar = csa_magpar(np.zeros(3), np.array([-1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
    assert abs(angle + np.pi / 2) < TOLERANCE, \
            "The solid angle is {}, but should be -PI/2={}.".format(angle, -np.pi/2)

    assert abs(abs(angle) - magpar) < TOLERANCE

def test_solid_angle_two_minus_signs():
    origin = np.zeros((3, 1))

    # (-, -, +) back-left-top
    triangle = np.array([[[-1.],[0.],[0.]], [[0.],[-1.],[0.]], [[0.],[0.],[1.]]])
    assert triangle.shape == (3, 3, 1)
    angle = csa(origin, triangle)[0]
    assert abs(angle - np.pi / 2) < TOLERANCE, \
            "The solid angle is {}, but should be PI/2={}.".format(angle[0], np.pi/2)

    magpar = csa_magpar(np.zeros(3), np.array([-1, 0, 0]), np.array([0, -1, 0]), np.array([0, 0, 1]))
    assert abs(angle - magpar) < TOLERANCE

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
    assert abs(angle) < TOLERANCE, \
        "The solid angle is {0}, but should be 0.".format(angle[0])

if __name__=="__main__":
    for name in globals().keys():
        if name.startswith("test_"):
            print "Running", name
            globals()[name]()
