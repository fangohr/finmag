import os
import dolfin
import numpy as np
import finmag.sim.helpers as h
from finmag.sim.llg import LLG
from scipy.integrate import ode

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 1e-1

# define the mesh
x_max = 100e-9 # m
simplexes = 50
mesh = dolfin.Interval(simplexes, 0, x_max)

def m_gen(coords):
  xs = coords[0]
  mx = np.minimum(np.ones(len(xs)), xs/x_max)
  mz = 0.1 * np.ones(len(xs))
  my = np.sqrt(1.0 - (0.99*mx**2 + mz*mz))
  return np.array([mx, my, mz])

coords = np.array(zip(* mesh.coordinates()))
m0 = m_gen(coords).flatten()

K1 = 520e3 # J/m^3

llg = LLG(mesh)
llg.Ms = 0.86e6
llg.alpha = 0.2
llg.set_m(m0)
llg.add_uniaxial_anisotropy(K1, dolfin.Constant((0, 0, 1)))
llg.setup(use_exchange=False)

t0 = 0; t1 = 3e-10; dt = 5e-12; # s
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
        tn_f.write(str(r.t) + " " + str(m2x) + " " + str(m2y) + " " + str(m2z) + "\n")

        r.integrate(r.t + dt)

    av_f.close()
    tn_f.close()

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
