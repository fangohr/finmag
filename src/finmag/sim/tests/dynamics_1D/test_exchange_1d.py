import sys
import numpy
import dolfin
import sys

from scipy.integrate import odeint, ode

from finmag.sim.llg import LLG
from finmag.sim.helpers import vectors,angle

#
# TODO: test with and without pinning, build comparison with
#       dynamics of nmag and then remove the file test_exchange_static
#

TOLERANCE = 1e-7

# define the mesh
length = 20e-9 #m
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

# initial configuration of the magnetisation
m0_x = '2*x[0]/L - 1'
m0_y = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'
m0_z = '0'

def test_exchange_without_pinning():
    llg = LLG(mesh)
    llg.set_m0((m0_x, m0_y, m0_z), L=length)
    llg.setup(exchange_flag=True)

    ts = numpy.linspace(0, 1e-9, 100)
    ys, infodict = odeint(llg.solve_for, llg.m, ts, full_output=True)

    m = vectors(ys[-1])
    angles = numpy.array([angle(m[j], m[j+1]) for j in xrange(len(m)-1)])
    assert abs(angles.max() - angles.min()) < TOLERANCE

def test_exchange_with_pinning():
    llg = LLG(mesh)
    llg.set_m0((m0_x, m0_y, m0_z), L=length)
    llg.setup(exchange_flag=True)
    llg.pins = [0, 10]

    ts = numpy.linspace(0, 1e-9, 100)
    ys, infodict = odeint(llg.solve_for, llg.m, ts, full_output=True)

    m = vectors(ys[-1])
    angles = numpy.array([angle(m[j], m[j+1]) for j in xrange(len(m)-1)])
    assert abs(angles.max() - angles.min()) < TOLERANCE

def _test_compare_averages():
    ref = [line.strip().split() for line in open("averages_ref.txt")]
    computed = [line.strip().split() for line in open("averages.txt")]

    for i in range(len(ref)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = computed[i]
        
        assert t_ref == t
        print "t={0}: ref={1}|{2}|{3} computed:{4}|{5}|{6}.".format(
                t, mx_ref, my_ref, mz_ref, mx, my, mz)
        assert abs(float(mx_ref) - float(mx)) < TOLERANCE
        assert abs(float(my_ref) - float(my)) < TOLERANCE
        assert abs(float(mz_ref) - float(mz)) < TOLERANCE

def _test_compare_third_node():
    ref = [line.strip().split() for line in open("third_node_ref.txt")]
    computed = [line.strip().split() for line in open("third_node.txt")]

    for i in range(len(ref)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = computed[i]
        
        assert t_ref == t
        print "t={0}: ref={1}|{2}|{3} computed:{4}|{5}|{6}.".format(
                t, mx_ref, my_ref, mz_ref, mx, my, mz)
        assert abs(float(mx_ref) - float(mx)) < TOLERANCE
        assert abs(float(my_ref) - float(my)) < TOLERANCE
        assert abs(float(mz_ref) - float(mz)) < TOLERANCE
