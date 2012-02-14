import numpy
import dolfin
from scipy.integrate import odeint

from finmag.sim.llg import LLG
from finmag.sim.helpers import make_vectors_function,angle

TOLERANCE = 5e-10

# define the mesh
length = 20e-9 #m
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

# initial configuration of the magnetisation
left_right = 'MS * (2*x[0]/L - 1)'
up_down = 'sqrt(MS*MS - MS*MS*(2*x[0]/L - 1)*(2*x[0]/L - 1))'

possible_orientations = [
    (left_right, up_down, '0'), # (left_right, '0', up_down),
    (up_down, '0', left_right), # (up_down, left_right, '0'),
    ('0', left_right, up_down)] # , ('0', up_down, left_right)]

def angles_after_a_nanosecond(initial_M, pins=[]):
    llg = LLG(mesh)
    llg.initial_M_expr(initial_M, L=length, MS=llg.MS)
    llg.setup()
    llg.pins = pins 

    ts = numpy.linspace(0, 1e-9, 2)
    ys, infodict = odeint(llg.solve_for, llg.M, ts, atol=10, full_output=True)

    vectors = make_vectors_function(ys[0])
    M = vectors(ys[-1])
    angles = numpy.array([angle(M[i], M[i+1]) for i in xrange(len(M)-1)])
    return angles

def test_all_orientations_without_pinning():
    for M0 in possible_orientations:
        angles = angles_after_a_nanosecond(M0)
        print angles
        assert angles.max() < TOLERANCE

def test_all_orientations_with_pinning():
    for M0 in possible_orientations:
        angles = angles_after_a_nanosecond(M0, [0, 10])
        print angles
        assert abs(angles.max() - angles.min()) < TOLERANCE

if __name__== "__main__":
    test_all_orientations_with_pinning()
    test_all_orientations_without_pinning()
