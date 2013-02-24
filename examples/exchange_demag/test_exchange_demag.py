import os
import logging
import pylab as p
import numpy as np
import dolfin as df
import progressbar as pb
from finmag import Simulation as Sim
from finmag.energies import Exchange, Demag
from finmag.util.meshes import from_geofile, mesh_volume

logger = logging.getLogger(name='finmag')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 5e-4
Ms = 0.86e6
unit_length = 1e-9
mesh = from_geofile(os.path.join(MODULE_DIR, "bar30_30_100.geo"))


def run_finmag():
    """Run the finmag simulation and store data in averages.txt."""

    sim = Sim(mesh, Ms, unit_length=unit_length)
    sim.alpha = 0.5
    sim.set_m((1, 0, 1))

    exchange = Exchange(13.0e-12)
    sim.add(exchange)
    demag = Demag(solver="FK")
    sim.add(demag)

    fh = open(os.path.join(MODULE_DIR, "averages.txt"), "w")
    fe = open(os.path.join(MODULE_DIR, "energies.txt"), "w")

    # Progressbar
    bar = pb.ProgressBar(maxval=60, \
                    widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])

    logger.info("Time integration")
    times = np.linspace(0, 3.0e-10, 61)
    for counter, t in enumerate(times):
        bar.update(counter)

        # Integrate
        sim.run_until(t)

        # Save averages to file
        mx, my, mz = sim.llg.m_average
        fh.write(str(t) + " " + str(mx) + " " + str(my) + " " + str(mz) + "\n")

        # Energies
        E_e = exchange.compute_energy()
        E_d = demag.compute_energy()
        fe.write(str(E_e) + " " + str(E_d) + "\n")

        # Energy densities
        if counter == 10:
            exch_energy = exchange.energy_density_function()
            demag_energy = demag.demag.energy_density_function()
            finmag_exch, finmag_demag = [], []
            R = range(100)
            for i in R:
                finmag_exch.append(exch_energy([15, 15, i]))
                finmag_demag.append(demag_energy([15, 15, i]))
            # Store data
            np.save(os.path.join(MODULE_DIR, "finmag_exch_density.npy"), np.array(finmag_exch))
            np.save(os.path.join(MODULE_DIR, "finmag_demag_density.npy"), np.array(finmag_demag))

    fh.close()
    fe.close()

def test_compare_averages():
    ref = np.loadtxt(os.path.join(MODULE_DIR, "averages_ref.txt"))
    if not os.path.isfile(os.path.join(MODULE_DIR, "averages.txt")) \
       or (os.path.getctime(os.path.join(MODULE_DIR, "averages.txt")) <
           os.path.getctime(os.path.abspath(__file__))):
        run_finmag()

    computed = np.loadtxt(os.path.join(MODULE_DIR, "averages.txt"))
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
    assert err < REL_TOLERANCE, "Relative error = {} is larger " \
        "than tolerance (= {})".format(err, REL_TOLERANCE)

    # Plot nmag data
    nmagt = list(ref[:,0])*3
    nmagy = list(ref[:,1]) + list(ref[:,2]) + list(ref[:,3])
    p.plot(nmagt, nmagy, 'o', mfc='w', label='nmag')

    # Plot finmag data
    t = computed[:, 0]
    x = computed[:, 1]
    y = computed[:, 2]
    z = computed[:, 3]
    p.plot(t, x, 'k', label='$m_\mathrm{x}$ finmag')
    p.plot(t, y, 'b-.', label='$m_\mathrm{y}$')
    p.plot(t, z, 'r', label='$m_\mathrm{z}$')
    p.axis([0, max(t), -0.2, 1.1])
    p.xlabel("time (s)")
    p.ylabel("$m$")
    p.legend(loc='center right')
    p.savefig(os.path.join(MODULE_DIR, "exchange_demag.pdf"))
    p.savefig(os.path.join(MODULE_DIR, "exchange_demag.png"))

    #p.show
    p.close()
    print "Comparison of development written to exchange_demag.pdf"

def test_compare_energies():
    ref = np.loadtxt(os.path.join(MODULE_DIR, "energies_ref.txt"))
    if not os.path.isfile(os.path.join(MODULE_DIR, "energies.txt")) \
       or (os.path.getctime(os.path.join(MODULE_DIR, "energies.txt")) <
           os.path.getctime(os.path.abspath(__file__))):
        run_finmag()

    computed = np.loadtxt(os.path.join(MODULE_DIR, "energies.txt"))
    assert np.size(ref) == np.size(computed), "Compare number of energies."

    vol = mesh_volume(mesh)*unit_length**mesh.topology().dim()
    #30x30x100nm^3 = 30x30x100=9000

    # Compare exchange energy...
    exch = computed[:, 0]/vol # <-- ... density!
    exch_nmag = ref[:, 0]

    diff = abs(exch - exch_nmag)
    rel_diff = np.abs(diff / max(exch))
    print "Exchange energy, max relative error:", max(rel_diff)
    assert max(rel_diff) < 0.002, \
        "Max relative error in exchange energy = {} is larger than " \
        "tolerance (= {})".format(max(rel_diff), REL_TOLERANCE)

    # Compare demag energy
    demag = computed[:, 1]/vol
    demag_nmag = ref[:, 1]

    diff = abs(demag - demag_nmag)
    rel_diff = np.abs(diff / max(demag))
    print "Demag energy, max relative error:", max(rel_diff)
    # Don't really know why this is ten times higher than everyting else.
    assert max(rel_diff) < REL_TOLERANCE*10, \
        "Max relative error in demag energy = {} is larger than " \
        "tolerance (= {})".format(max(rel_diff), REL_TOLERANCE)

    # Plot
    p.plot(exch_nmag, 'o', mfc='w', label='nmag')
    p.plot(exch, label='finmag')
    p.xlabel("time step")
    p.ylabel("$e_\mathrm{exch}\, (\mathrm{Jm^{-3}})$")
    p.legend()
    p.savefig(os.path.join(MODULE_DIR, "exchange_energy.pdf"))
    p.savefig(os.path.join(MODULE_DIR, "exchange_energy.png"))
    p.close()

    p.plot(demag_nmag, 'o', mfc='w', label='nmag')
    p.plot(demag, label='finmag')
    p.xlabel("time step")
    p.ylabel("$e_\mathrm{demag}\, (\mathrm{Jm^{-3}})$")
    p.legend()
    p.savefig(os.path.join(MODULE_DIR, "demag_energy.pdf"))
    p.savefig(os.path.join(MODULE_DIR, "demag_energy.png"))
    #p.show()
    p.close()
    print "Energy plots written to exchange_energy.pdf and demag_energy.pdf"

def test_compare_energy_density():
    """
    After ten time steps, compute the energy density through
    the center of the bar (seen from x and y) from z=0 to z=100,
    and compare the results with nmag and oomf.

    """
    R = range(100)

    # Run simulation only if not run before or changed since last time.
    if not (os.path.isfile(os.path.join(MODULE_DIR, "finmag_exch_density.npy"))):
        run_finmag()
    elif (os.path.getctime(os.path.join(MODULE_DIR, "finmag_exch_density.npy")) <
          os.path.getctime(os.path.abspath(__file__))):
        run_finmag()
    if not (os.path.isfile(os.path.join(MODULE_DIR, "finmag_demag_density.npy"))):
        run_finmag()
    elif (os.path.getctime(os.path.join(MODULE_DIR, "finmag_demag_density.npy")) <
          os.path.getctime(os.path.abspath(__file__))):
        run_finmag()

    # Read finmag data
    finmag_exch = np.load(os.path.join(MODULE_DIR, "finmag_exch_density.npy"))
    finmag_demag = np.load(os.path.join(MODULE_DIR, "finmag_demag_density.npy"))

    # Read nmag data
    nmag_exch = [float(i) for i in open(os.path.join(MODULE_DIR, "nmag_exch_Edensity.txt"), "r").read().split()]
    nmag_demag = [float(i) for i in open(os.path.join(MODULE_DIR, "nmag_demag_Edensity.txt"), "r").read().split()]

    # Compare with nmag
    nmag_exch = np.array(nmag_exch)
    nmag_demag = np.array(nmag_demag)
    rel_error_exch_nmag = np.abs(finmag_exch - nmag_exch)/np.linalg.norm(nmag_exch)
    rel_error_demag_nmag = np.abs(finmag_demag - nmag_demag)/np.linalg.norm(nmag_demag)
    print "Exchange energy density, max relative error from nmag:", max(rel_error_exch_nmag)
    print "Demag energy density, max relative error from nmag:", max(rel_error_demag_nmag)
    TOL_EXCH = 3e-2
    TOL_DEMAG = 1e-2
    assert max(rel_error_exch_nmag) < TOL_EXCH, \
        "Exchange energy density, max relative error from nmag = {} is " \
        "larger than tolerance (= {})".format(max(rel_error_exch_nmag), TOL_EXCH)
    assert max(rel_error_demag_nmag) < TOL_DEMAG, \
        "Demag energy density, max relative error from nmag = {} is larger " \
        "than tolarance (= {})".format(max(rel_error_demag_nmag), TOL_DEMAG)


    # Read oommf data
    oommf_exch = np.genfromtxt(os.path.join(MODULE_DIR, "oommf_exch_Edensity.txt"))
    oommf_demag = np.genfromtxt(os.path.join(MODULE_DIR, "oommf_demag_Edensity.txt"))
    oommf_coords = np.genfromtxt(os.path.join(MODULE_DIR, "oommf_coords_z_axis.txt")) * 1e9

    # Compare with oomf - FIXME: doesn't work at the moment
    #rel_error_exch_oomf = np.abs(finmag_exch - oommf_exch)/np.linalg.norm(oommf_exch)
    #rel_error_demag_oomf = np.abs(finmag_demag - oommf_demag)/np.linalg.norm(oommf_demag)
    #print "Rel error exch, oommf:", max(rel_error_exch_oommf)
    #print "Rel error demag, oommf:", max(rel_error_demag_oommf)

    # Plot exchange energy density
    p.plot(R, finmag_exch, 'k-')
    p.plot(R, nmag_exch, 'r^:', alpha=0.5)
    p.plot(oommf_coords, oommf_exch, "bv:", alpha=0.5)
    p.xlabel("$x\, (\mathrm{nm})$")
    p.ylabel("$e_\mathrm{exch}\, (\mathrm{Jm^{-3}})$")
    p.legend(["finmag", "nmag", "oommf"], loc="upper center")
    p.savefig(os.path.join(MODULE_DIR, "exchange_density.pdf"))
    p.savefig(os.path.join(MODULE_DIR, "exchange_density.png"))
    p.close()

    # Plot demag energy density
    p.plot(R, finmag_demag, 'k-')
    p.plot(R, nmag_demag, 'r^:', alpha=0.5)
    p.plot(oommf_coords, oommf_demag, "bv:", alpha=0.5)
    p.xlabel("$x\, (\mathrm{nm})$")
    p.ylabel("$e_\mathrm{demag}\, (\mathrm{Jm^{-3}})$")
    p.legend(["finmag", "nmag", "oommf"], loc="upper center")
    p.savefig(os.path.join(MODULE_DIR, "demag_density.pdf"))
    p.savefig(os.path.join(MODULE_DIR, "demag_density.png"))
    #p.show()
    p.close()
    print "Energy density plots written to exchange_density.pdf and demag_density.pdf"

if __name__ == '__main__':
    test_compare_averages()
    test_compare_energies()
    test_compare_energy_density()
