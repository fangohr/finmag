import os
from finmag.sim.llg import LLG
from scipy.integrate import ode
from finmag.util.convert_mesh import convert_mesh
import pytest
import pylab as p
import numpy as np
import dolfin as df
import progressbar as pb
import finmag.sim.helpers as h

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 3e-2

def save_plot(t, x, y, z):
    """Save plot of finmag data and comparisson with nmag data (if exist)."""

    # Add data points from nmag to plot
    if os.path.isfile(MODULE_DIR + "/averages_ref.txt"):
        ref = np.array(h.read_float_data(MODULE_DIR + "/averages_ref.txt"))
        dt = ref[:,0] - np.array(t)
        assert np.max(dt) < 1e-15, "Compare timesteps."
        nmagt = list(ref[:,0])*3
        nmagy = list(ref[:,1]) + list(ref[:,2]) + list(ref[:,3])
        p.plot(nmagt, nmagy, 'o', mfc='w', label='nmag')

    # Plot finmag data
    p.plot(t, x, 'k', label='$\mathsf{m_x}$')
    p.plot(t, y, 'r', label='$\mathsf{m_y}$')
    p.plot(t, z, 'b', label='$\mathsf{m_z}$')
    p.axis([0, max(t), -0.2, 1.1])
    p.title("Finmag vs Nmag")
    p.legend(loc='center right')
    p.savefig(MODULE_DIR + "/exchange_demag.png")

def run_finmag():
    """Run the finmag simulation and store data in averages.txt."""
    mesh = df.Mesh(convert_mesh(MODULE_DIR + "/bar30_30_100.geo"))

    # Set up LLG
    llg = LLG(mesh, mesh_units=1e-9)
    llg.Ms = 0.86e6
    llg.A = 13.0e-12
    llg.alpha = 0.5
    llg.set_m((1,0,1))
    llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="weiwei")

    # Set up time integrator
    llg_wrap = lambda t, y: llg.solve_for(y, t)
    r = ode(llg_wrap).set_integrator("vode", method="bdf", rtol=1e-5, atol=1e-5)
    r.set_initial_value(llg.m, 0)

    dt = 5.0e-12
    T1 = 60*dt

    # Progressbar
    fh = open(MODULE_DIR + "/averages.txt", "w")
    xlist, ylist, zlist, tlist = [], [], [], []

    bar = pb.ProgressBar(maxval=60, \
                    widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])
    counter = 0
    print "Time integration (this may take some time..)"
    while r.successful() and r.t <= T1:
        bar.update(counter)
        counter += 1

        mx, my, mz = llg.m_average
        xlist.append(mx)
        ylist.append(my)
        zlist.append(mz)
        tlist.append(r.t)

        fh.write(str(r.t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")
        r.integrate(r.t + dt)
    fh.close()
    save_plot(tlist, xlist, ylist, zlist)

#@pytest.mark.skipif("not os.path.exists(MODULE_DIR + '/averages.txt')")
def test_compare_averages():
    ref = np.array(h.read_float_data(MODULE_DIR + "/averages_ref.txt"))
    if not (os.path.isfile(MODULE_DIR + "/averages.txt") and
            os.path.isfile(MODULE_DIR + "/exchange_demag.png")):
        run_finmag()

    computed = np.array(h.read_float_data(MODULE_DIR + "/averages.txt"))
    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref, computed = np.delete(ref, [0], 1), np.delete(computed, [0], 1)
    diff = ref - computed
    rel_diff = np.abs(diff / np.sqrt(ref[0]**2 + ref[1]**2 + ref[2]**2))

    print "test_averages, max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)

    err = np.nanmax(rel_diff)
    #if err > 1e-3:
    if err > 2.5e-2:
        print "nmag:\n", ref
        print "finmag:\n", computed
    assert err < REL_TOLERANCE

if __name__ == '__main__':
    test_compare_averages()
