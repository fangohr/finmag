import os
import numpy as np
import finmag.util.helpers as h
from dolfin import IntervalMesh
from finmag import Simulation as Sim
from finmag.energies import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def setup_module(module=None):
    # define the mesh
    x_max = 20e-9 # m
    simplexes = 10
    mesh = IntervalMesh(simplexes, 0, x_max)

    def m_gen(coords):
        x = coords[0]
        mx = min(1.0, 2.0 * x/x_max - 1.0)
        my = np.sqrt(1.0 - mx**2)
        mz = 0.0
        return np.array([mx, my, mz])

    Ms = 0.86e6
    A = 1.3e-11

    global sim
    sim = Sim(mesh, Ms)
    sim.alpha = 0.2
    sim.set_m(m_gen)
    sim.pins = [0, 10]
    exchange = Exchange(A)
    sim.add(exchange)

    # Save H_exc and m at t0 for comparison with nmag
    global H_exc_t0, m_t0
    H_exc_t0 = exchange.compute_field()
    m_t0 = sim.m

    t = 0; t1 = 5e-10; dt = 1e-11; # s

    av_f = open(os.path.join(MODULE_DIR, "averages.txt"), "w")
    tn_f = open(os.path.join(MODULE_DIR, "third_node.txt"), "w")

    global averages
    averages = []
    global third_node
    third_node = []

    while t <= t1:
        mx, my, mz = sim.m_average
        averages.append([t, mx, my, mz])
        av_f.write(str(t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

        mx, my, mz = h.components(sim.m)
        m2x, m2y, m2z = mx[2], my[2], mz[2]
        third_node.append([t, m2x, m2y, m2z])
        tn_f.write(str(t) + " " + str(m2x) + " " + str(m2y) + " " + str(m2z) + "\n")

        t += dt
        sim.run_until(t)

    av_f.close()
    tn_f.close()

def test_angles():
    TOLERANCE = 5e-8

    m = h.vectors(sim.m)
    angles = np.array([h.angle(m[i], m[i+1]) for i in xrange(len(m)-1)])

    max_diff = abs(angles.max() - angles.min())
    mean_angle = np.mean(angles)
    print "test_angles: max_difference= {}.".format(max_diff)
    print "test_angles: mean= {}.".format(mean_angle)
    assert max_diff < TOLERANCE
    assert np.abs(mean_angle - np.pi/10) < TOLERANCE

def test_averages():
    TOLERANCE = 2e-3
    """
    We compare absolute values here, because values which should be
    exactly zero in the idealised physical experiment (z-components of the
    magnetisation as well as the average of the x-component) are not numerically.

    In nmag, these "zeros" have the order of magnitude 1e-8, whereas
    in finmag, they are in the order of 1e-14 and less. The difference is
    roughly 1e-8 and the relative difference (dividing by nmag) would be 1.

    That's useless for comparing. Solutions:
    1. compute the relative difference by dividing by the norm of the vector or
    something like this. Meh...
    2. Check for zeros instead of comparing with nmag. But then you couldn't
    copy&paste the comparison code anymore.
    3. Write this comment and compare absolute values. Note that the tolerance
    reflects the difference beetween non-zero components.

    """
    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = np.array(averages)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    print "test_averages, max. difference per axis:"
    print np.nanmax(np.abs(diff), axis=0)

    assert np.nanmax(diff) < TOLERANCE

def test_third_node():
    REL_TOLERANCE = 6e-3

    ref = np.loadtxt(os.path.join(MODULE_DIR, "third_node_ref.txt"))
    computed = np.array(third_node)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / ref)

    print "test_third_node, max. difference per axis:"
    print np.nanmax(np.abs(diff), axis=0)
    print "test_third_node, max. relative difference per axis:"
    max_diffs = np.nanmax(rel_diff, axis=0)
    print max_diffs
    assert max_diffs[0] < REL_TOLERANCE and max_diffs[1] < REL_TOLERANCE

def test_m_cross_H():
    """
    compares m x H_exc at the beginning of the simulation.

    """
    REL_TOLERANCE = 8e-8

    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m_t0_ref.txt"))
    m_computed = h.vectors(m_t0)
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "exc_t0_ref.txt"))
    H_computed = h.vectors(H_exc_t0)
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    diff = np.abs(m_cross_H_ref - m_cross_H_computed)
    max_norm = max([h.norm(v) for v in m_cross_H_ref])
    rel_diff = diff/max_norm

    print "test_m_cross_H, max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    setup_module()
    test_angles()
    test_averages()
    test_third_node()
    test_m_cross_H()
