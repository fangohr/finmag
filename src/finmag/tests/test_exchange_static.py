import numpy as np
import dolfin as df
from finmag import Simulation as Sim
from finmag.energies import Exchange
from finmag.util.helpers import vectors, angle

TOLERANCE = 8e-7

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

def test_exchange_field_should_change_when_M_changes():
    sim = Sim(mesh, Ms)
    sim.set_m(df.Expression(
        ('(2*x[0]-L)/L',
         'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
         '0'), L=length))

    exchange = Exchange(A)
    sim.add(exchange)

    # save the beginning value of M and the exchange field for comparison purposes
    old_m = sim.m
    old_H_ex = exchange.compute_field()

    sim.run_until(1e-11)

    # Capture the current value of the exchange field and m.
    m = sim.m
    H_ex = exchange.compute_field()

    # We assert that the magnetisation has indeed changed since the beginning.
    assert not np.array_equal(old_m, m)
    assert not np.array_equal(old_H_ex, H_ex), "H_ex hasn't changed."

if __name__== "__main__":
    print "without pinning"
    test_all_orientations_without_pinning()
    print "with pinning"
    test_all_orientations_with_pinning()
