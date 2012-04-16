import os
import numpy as np
import finmag.sim.helpers as h
from dolfin import Interval
from finmag.sim.llg import LLG
from scipy.integrate import ode

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_module(module=None):
    # define the mesh
    x_max = 60e-9 # m
    simplexes = 30
    mesh = Interval(simplexes, 0, x_max)

    def m_gen(coords):
        xs = coords[0]
        mx = np.minimum(np.ones(len(xs)), 2.0 * xs/x_max - 1.0)
        my = np.sqrt(1.0 - mx**2)
        mz = np.zeros(len(xs))
        return np.array([mx, my, mz])
    coords = np.array(zip(* mesh.coordinates()))
    m0 = m_gen(coords).flatten()

    global llg
    llg = LLG(mesh)
    llg.Ms = 0.86e6
    llg.A = 1.3e-11
    llg.alpha = 0.2
    llg.set_m(m0)
    llg.setup(use_exchange=True)
    llg.pins = [0, 30]

    t0 = 0; t1 = 1e-10; dt = 1e-12; # s
    # ode takes the parameters in the order t, y whereas odeint and we use y, t.
    llg_wrap = lambda t, y: llg.solve_for(y, t)
    r = ode(llg_wrap).set_integrator("vode", method="bdf")
    r.set_initial_value(llg.m, t0)

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

def test_angles():
    TOLERANCE = 4e-2

    m = h.vectors(llg.m)
    angles = np.array([h.angle(m[i], m[i+1]) for i in xrange(len(m)-1)])

    max_diff = abs(angles.max() - angles.min())
    print "test_angles: max_difference= {}.".format(max_diff)
    assert max_diff < TOLERANCE

def test_averages():
    REL_TOLERANCE = 1.001

    ref = np.array(h.read_float_data(MODULE_DIR + "/averages_ref.txt"))
    computed = np.array(averages)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / ref)

    print "test_averages, max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)

    err = np.nanmax(rel_diff)
    if err > 1e-3:
        print "nmag:\n", ref
        print "finmag:\n", computed
    assert np.nanmax(rel_diff) < REL_TOLERANCE

def test_third_node():
    REL_TOLERANCE = 2e-1

    ref = np.array(h.read_float_data(MODULE_DIR + "/third_node_ref.txt"))
    computed = np.array(third_node)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / ref)

    print "test_third_node, max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)
    assert np.nanmax(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    setup_module()
    test_angles()
    test_averages()
    test_third_node()
