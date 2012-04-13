import pytest
import os
import dolfin
import numpy as np
import finmag.sim.helpers as h
from finmag.sim.llg import LLG
from scipy.integrate import ode

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

averages = []
third_node = []

# run the simulation
def setup_module(module=None):
    x_max = 20e-9 # m
    simplexes = 10
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

    # Save H_anis at t0 for comparison with nmag
    global H_anis
    H_anis = llg._anisotropies[0].compute_field()

    t0 = 0; t1 = 3e-10; dt = 5e-12; # s
    # ode takes the parameters in the order t, y whereas odeint and we use y, t.
    llg_wrap = lambda t, y: llg.solve_for(y, t)
    r = ode(llg_wrap).set_integrator("vode", method="bdf")
    r.set_initial_value(llg.m, t0)

    av_f = open(MODULE_DIR + "/averages.txt", "w")
    tn_f = open(MODULE_DIR + "/third_node.txt", "w")

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

def test_averages():
    TOLERANCE = 5e-2

    ref = np.array(h.read_float_data(MODULE_DIR + "/averages_ref.txt"))
    computed = np.array(averages)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    diff = np.delete(np.abs(ref - computed), [0], 1) # diff without time information
    assert np.max(diff) < TOLERANCE

def test_third_node():
    TOLERANCE = 4e-2

    ref = np.array(h.read_float_data(MODULE_DIR + "/third_node_ref.txt"))
    computed = np.array(third_node)

    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    diff = np.delete(np.abs(ref - computed), [0], 1) # diff without time information
    assert np.max(diff) < TOLERANCE

@pytest.mark.xfail
def test_anisotropy_field():
    REL_TOLERANCE = 1e-3

    ref = np.genfromtxt(MODULE_DIR + "/anis_t0_ref.txt")
    computed = h.vectors(H_anis)

    diff = np.abs(computed - ref)
    rel_diff = diff / ref
    print "Expected:\n", ref, "\ngot:\n", computed

    print "Comparing the anisotropy fields of finmag and nmag."
    print "The maximum relative difference should be smaller than 1e-3."
    assert np.nanmax(np.abs(rel_diff)) < REL_TOLERANCE 

if __name__ == '__main__':
    setup_module()
    test_anisotropy_field() 
