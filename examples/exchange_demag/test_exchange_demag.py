import os
import pytest
import logging
import pylab as p
import numpy as np
import dolfin as df
import progressbar as pb
import finmag.sim.helpers as h
from finmag.sim.llg import LLG
from finmag.sim.integrator import LLGIntegrator
from finmag.util.convert_mesh import convert_mesh

logger = logging.getLogger(name='finmag')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 1e-4

unit_length = 1e-9
mesh = df.Mesh(convert_mesh(MODULE_DIR + "/bar30_30_100.geo"))

def run_finmag():
    """Run the finmag simulation and store data in averages.txt."""

    # Set up LLG
    llg = LLG(mesh, unit_length=unit_length)
    llg.Ms = 0.86e6
    llg.A = 13.0e-12
    llg.alpha = 0.5
    llg.set_m((1,0,1))
    llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="FK")

    # Set up time integrator
    integrator = LLGIntegrator(llg, llg.m)

    fh = open(MODULE_DIR + "/averages.txt", "w")
    fe = open(MODULE_DIR + "/energies.txt", "w")

    # Progressbar
    bar = pb.ProgressBar(maxval=60, \
                    widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])

    logger.info("Time integration")
    times = np.linspace(0, 3.0e-10, 61)
    for counter, t in enumerate(times):
        bar.update(counter)

        # Integrate
        integrator.run_until(t)

        # Save averages to file
        mx, my, mz = llg.m_average
        fh.write(str(t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

        # Energies
        E_e = llg.exchange.compute_energy()
        E_d = llg.demag.compute_energy()
        fe.write(str(E_e) + " " + str(E_d) + "\n")

    fh.close()
    fe.close()

def test_compare_averages():
    ref = np.array(h.read_float_data(MODULE_DIR + "/averages_ref.txt"))
    if not os.path.isfile(MODULE_DIR + "/averages.txt"):
        run_finmag()
    elif (os.path.getctime(MODULE_DIR + "/averages.txt") <
          os.path.getctime(os.path.abspath(__file__))):
        run_finmag()

    computed = np.array(h.read_float_data(MODULE_DIR + "/averages.txt"))
    dt = ref[:,0] - computed[:,0]
    assert np.max(dt) < 1e-15, "Compare timesteps."

    ref1, computed1 = np.delete(ref, [0], 1), np.delete(computed, [0], 1)

    diff = ref1 - computed1
    print "max difference: %g" % np.max(diff)

    rel_diff = np.abs(diff / np.sqrt(ref1[0]**2 + ref1[1]**2 + ref1[2]**2))
    print "test_averages, max. relative difference per axis:"
    print np.nanmax(rel_diff, axis=0)

    err = np.nanmax(rel_diff)
    if err > 1e-2:
        print "nmag:\n", ref1
        print "finmag:\n", computed1
    assert err < REL_TOLERANCE, "Relative error = %g" % err

    # Plot nmag data
    nmagt = list(ref[:,0])*3
    nmagy = list(ref[:,1]) + list(ref[:,2]) + list(ref[:,3])
    p.plot(nmagt, nmagy, 'o', mfc='w', label='nmag')

    # Plot finmag data
    t = computed[:, 0]
    x = computed[:, 1]
    y = computed[:, 2]
    z = computed[:, 3]
    p.plot(t, x, 'k', label='$\mathsf{m_x}$')
    p.plot(t, y, 'r', label='$\mathsf{m_y}$')
    p.plot(t, z, 'b', label='$\mathsf{m_z}$')
    p.axis([0, max(t), -0.2, 1.1])
    p.xlabel("Time")
    p.ylabel("m")
    p.title("Finmag vs Nmag")
    p.legend(loc='center right')
    p.savefig(MODULE_DIR + "/exchange_demag.png")

def test_compare_energies():
    ref = np.array(h.read_float_data(MODULE_DIR + "/energies_ref.txt"))
    if not (os.path.isfile(MODULE_DIR + "/energies.txt")):
        run_finmag()
    elif (os.path.getctime(MODULE_DIR + "/energies.txt") <
          os.path.getctime(os.path.abspath(__file__))):
        run_finmag()

    computed = np.array(h.read_float_data(MODULE_DIR + "/energies.txt"))
    assert np.size(ref) == np.size(computed), "Compare number of energies."

    vol = df.assemble(df.Constant(1)*df.dx, mesh=mesh)*unit_length**mesh.topology().dim()
    #30x30x100nm^3 = 30x30x100=9000

    # Compare exchange energy
    exch = computed[:, 0]/vol
    exch_nmag = ref[:, 0]

    diff = abs(exch - exch_nmag)
    rel_diff = np.abs(diff / max(exch))
    print "Exchange energy, max relative error:", max(rel_diff)
    assert max(rel_diff) < REL_TOLERANCE, \
            "Max relative error in exchange energy is %g" % max(rel_diff)

    # Compare demag energy
    demag = computed[:, 1]/vol
    demag_nmag = ref[:, 1]

    diff = abs(demag - demag_nmag)
    rel_diff = np.abs(diff / max(demag))
    print "Demag energy, max relative error:", max(rel_diff)
    # Don't really know why this is ten times higher than everyting else.
    assert max(rel_diff) < REL_TOLERANCE*10, \
            "Max relative error in demag energy is %g" % max(rel_diff)

    # Plot
    p.figure()
    p.plot(exch_nmag, 'o', mfc='w', label='Nmag')
    p.plot(exch, label='Finmag')
    p.xlabel("Time step")
    p.ylabel("$\mathsf{E_{exch}}$")
    p.legend()
    p.savefig(MODULE_DIR + "/exchange_energy.png")
    p.figure()
    p.plot(demag_nmag, 'o', mfc='w', label='Nmag')
    p.plot(demag, label='Finmag')
    p.xlabel("Time step")
    p.ylabel("$\mathsf{E_{demag}}$")
    p.legend()
    p.savefig(MODULE_DIR + "/demag_energy.png")

if __name__ == '__main__':
    test_compare_averages()
    test_compare_energies()
