import os
import dolfin
import numpy
import finmag.sim.helpers as h
from finmag.sim.llg import LLG
from scipy.integrate import ode

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 4e-1

# define the mesh
length = 20e-9 # m
simplexes = 10
mesh = dolfin.Interval(simplexes, 0, length)

# initial configuration of the magnetisation
m0_x = '2*x[0]/L - 1'
m0_y = 'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))'
m0_z = '0'

llg = LLG(mesh)
llg.Ms = 0.86e6
llg.A = 1.3e-11
llg.alpha = 0.2
llg.set_m((m0_x, m0_y, m0_z), L=length)
llg.setup(use_exchange=True)
llg.pins = [0, 10]

t0 = 0; t1 = 1e-10; dt = 1e-12; # s
# ode takes the parameters in the order t, y whereas odeint and we use y, t.
llg_wrap = lambda t, y: llg.solve_for(y, t)
r = ode(llg_wrap).set_integrator("vode", method="bdf")
r.set_initial_value(llg.m, t0)

# run the simulation
def setup_module(module):
    av_f = open(MODULE_DIR + "/averages.txt", "w")
    tn_f = open(MODULE_DIR + "/third_node.txt", "w")

    global averages
    averages = []
    global third_node
    third_node = []

    while r.successful() and r.t <= t1:
        mx, my, mz = llg.m_average
        averages.append([r.t, mx, my, mz])
        av_f.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

        mx, my, mz = h.components(llg.m)
        m2x, m2y, m2z = mx[2], my[2], mz[2]
        third_node.append([r.t, m2x, m2y, m2z])
        tn_f.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

        r.integrate(r.t + dt)

    av_f.close()
    tn_f.close()

def test_angles():
    m = h.vectors(llg.m)
    angles = numpy.array([h.angle(m[i], m[i+1]) for i in xrange(len(m)-1)])
    assert abs(angles.max() - angles.min()) < TOLERANCE

def test_compare_averages():
    ref = [line.strip().split() for line in open(MODULE_DIR + "/averages_ref.txt")]

    for i in range(len(ref)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = averages[i]
        
        assert abs(float(t_ref) - t) < dt/2 
        print "t={0}: ref={1}|{2}|{3} computed:{4}|{5}|{6}.".format(
                t, mx_ref, my_ref, mz_ref, mx, my, mz)
        assert abs(float(mx_ref) - mx) < TOLERANCE
        assert abs(float(my_ref) - my) < TOLERANCE
        assert abs(float(mz_ref) - mz) < TOLERANCE

def test_compare_third_node():
    ref = [line.strip().split() for line in open(MODULE_DIR + "/third_node_ref.txt")]

    for i in range(len(ref)):
        t_ref, mx_ref, my_ref, mz_ref = ref[i]
        t, mx, my, mz = third_node[i]
        
        assert abs(float(t_ref) - t) < dt/2
        print "t={0}: ref={1}|{2}|{3} computed:{4}|{5}|{6}.".format(
                t, mx_ref, my_ref, mz_ref, mx, my, mz)
        assert abs(float(mx_ref) - mx) < TOLERANCE
        assert abs(float(my_ref) - my) < TOLERANCE
        assert abs(float(mz_ref) - mz) < TOLERANCE
