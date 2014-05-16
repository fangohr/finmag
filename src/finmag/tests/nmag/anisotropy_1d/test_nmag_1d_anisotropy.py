import os
import numpy as np
import finmag.util.helpers as h
from finmag import Simulation as Sim
from finmag.energies import UniaxialAnisotropy
import dolfin

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

averages = []
third_node = []

# run the simulation
def setup_module(module=None):
    x_max = 100e-9 # m
    simplexes = 50
    mesh = dolfin.IntervalMesh(simplexes, 0, x_max)

    def m_gen(coords):
        x = coords[0]
        mx = min(1.0, x/x_max)
        mz = 0.1
        my = np.sqrt(1.0 - (0.99*mx**2 + mz**2))
        return np.array([mx, my, mz]) 


    K1 = 520e3 # J/m^3
    Ms = 0.86e6

    sim = Sim(mesh, Ms)
    sim.alpha = 0.2
    sim.set_m(m_gen)
    anis = UniaxialAnisotropy(K1, (0, 0, 1))
    sim.add(anis)

    # Save H_anis and m at t0 for comparison with nmag
    global H_anis_t0, m_t0
    H_anis_t0 = anis.compute_field()
    m_t0 = sim.m


    av_f = open(os.path.join(MODULE_DIR, "averages.txt"), "w")
    tn_f = open(os.path.join(MODULE_DIR, "third_node.txt"), "w")

    t = 0; t_max = 3e-10; dt = 5e-12; # s
    while t <= t_max:
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

def test_averages():
    REL_TOLERANCE = 9e-2

    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    computed = np.array(averages)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / np.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2))

    print "test_averages, max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)

    rel_err = np.nanmax(rel_diff)
    if rel_err > 1e-3:
        print "nmag:\n", ref
        print "finmag:\n", computed
    assert rel_err < REL_TOLERANCE

def test_third_node():
    REL_TOLERANCE = 3e-1

    ref = np.loadtxt(os.path.join(MODULE_DIR, "third_node_ref.txt"))
    computed = np.array(third_node)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / np.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2))
   
    print "test_third_node: max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)

    rel_err = np.nanmax(rel_diff)
    if rel_err > 1e-3:
        print "nmag:\n", ref
        print "finmag:\n", computed
    assert rel_err < REL_TOLERANCE

def test_m_cross_H():
    """
    compares m x H_anis at the beginning of the simulation.
    motivation: Hans on IRC, 13.04.2012 10:45

    """ 
    REL_TOLERANCE = 7e-5

    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m_t0_ref.txt"))
    m_computed = h.vectors(m_t0)
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "anis_t0_ref.txt"))
    H_computed = h.vectors(H_anis_t0)
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    diff = np.abs(m_cross_H_ref - m_cross_H_computed)
    max_norm = max([h.norm(v) for v in m_cross_H_ref])
    rel_diff = diff/max_norm
  
    print "test_m_cross_H: max rel diff=",np.max(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    setup_module()
    test_averages()
    test_third_node()
    test_m_cross_H() 
