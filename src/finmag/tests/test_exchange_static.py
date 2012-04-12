import numpy
import dolfin
from scipy.integrate import odeint

from finmag.sim.llg import LLG
from finmag.sim.helpers import vectors,angle

TOLERANCE = 3e-9

# define the mesh
length = 20e-9 #m
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

# initial configuration of the magnetisation
left_right = '2*x[0]/L - 1'
up_down = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'

possible_orientations = [
    (left_right, up_down, '0'), # (left_right, '0', up_down),
    (up_down, '0', left_right), # (up_down, left_right, '0'),
    ('0', left_right, up_down)] # , ('0', up_down, left_right)]

def angles_after_a_nanosecond(initial_M, pins=[]):
    llg = LLG(mesh)
    llg.set_m(initial_M, L=length)
    llg.setup()
    llg.pins = pins 

    ts = numpy.linspace(0, 1e-9, 2)
    ys, infodict = odeint(llg.solve_for, llg.m, ts, rtol=1e-4, full_output=True)

    m = vectors(ys[-1])
    angles = numpy.array([angle(m[i], m[i+1]) for i in xrange(len(m)-1)])
    return angles

def test_all_orientations_without_pinning():
    for m0 in possible_orientations:
        angles = angles_after_a_nanosecond(m0)
        print angles
        assert angles.max() < TOLERANCE

def test_all_orientations_with_pinning():
    for m0 in possible_orientations:
        angles = angles_after_a_nanosecond(m0, [0, 10])
        print angles
        assert abs(angles.max() - angles.min()) < TOLERANCE

if __name__== "__main__":
    print "without pinning"
    test_all_orientations_without_pinning()
    print "with pinning"
    test_all_orientations_with_pinning()
