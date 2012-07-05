import numpy as np
import dolfin as df
from finmag import Simulation as Sim
from finmag.energies import Exchange
from finmag.util.helpers import vectors, angle

TOLERANCE = 1e-8

# define the mesh
length = 20e-9 #m
simplexes = 10
mesh = df.Interval(simplexes, 0, length)
Ms = 8.6e5
A = 1.3e-11

# initial configuration of the magnetisation
left_right = '2*x[0]/L - 1'
up_down = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'

possible_orientations = [
    (left_right, up_down, '0'), # (left_right, '0', up_down),
    (up_down, '0', left_right)] #, (up_down, left_right, '0'),
    #('0', left_right, up_down)] , ('0', up_down, left_right)]

def angles_after_a_nanosecond(initial_M, pins=[]):
    sim = Sim(mesh, Ms)
    sim.set_m(initial_M, L=length)
    sim.add(Exchange(A))
    sim.pins = pins 
    sim.run_until(1e-9)

    m = vectors(sim.m)
    angles = np.array([angle(m[i], m[i+1]) for i in xrange(len(m)-1)])
    return angles

def test_all_orientations_without_pinning():
    for m0 in possible_orientations:
        angles = angles_after_a_nanosecond(m0)
        print "no pinning, all angles: "
        print angles
        assert np.nanmax(angles) < TOLERANCE

def test_all_orientations_with_pinning():
    for m0 in possible_orientations:
        angles = angles_after_a_nanosecond(m0, [0, 10])
        print "no pinning, all angles: "
        print angles
        assert np.abs(np.max(angles) - np.min(angles)) < TOLERANCE

if __name__== "__main__":
    print "without pinning"
    test_all_orientations_without_pinning()
    print "with pinning"
    test_all_orientations_with_pinning()
