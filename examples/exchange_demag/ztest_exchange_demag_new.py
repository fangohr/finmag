""" What is this file? It appears that Gabrial has taken the test_exchange_demag.py and 
tried to add the GCR solver for the nmag 2 example. The test fails as the system seems to 
hang somewhere.

He then copied that file from test_exchange_demag.py to ztest_exchange_demag_new.py. Note
the z in the beginning which preventst the file from being run automatically as a 
regression test.

It would be good to understand at some point why the GCR solver hangs, but it also
be good to many other things.

Hans, 26 June 5:30am, somewhere over Malaysia.

"""


import os
import logging
import pylab as p
import numpy as np
import dolfin as df
import progressbar as pb
import finmag.util.helpers as h
from finmag import Simulation as Sim
from finmag.energies import Exchange, Demag
from finmag.util.meshes import from_geofile, mesh_volume

logger = logging.getLogger(name='finmag')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 1e-4
Ms = 0.86e6
unit_length = 1e-9
mesh = from_geofile(os.path.join(MODULE_DIR, "bar30_30_100.geo"))

demagsolvers = ["GCR"]


def run_finmag(demagsolver):
    """Run the finmag simulation and store data in (demagsolvertype)averages.txt."""

    sim = Sim(mesh, Ms, unit_length=unit_length)
    sim.alpha = 0.5
    sim.set_m((1, 0, 1))

    exchange = Exchange(13.0e-12)
    sim.add(exchange)
    demag = Demag(solver=demagsolver)
    sim.add(demag)

    fh = open(os.path.join(MODULE_DIR, demagsolver+"averages.txt"), "w")
    fe = open(os.path.join(MODULE_DIR, demagsolver+"energies.txt"), "w")

    # Progressbar
    bar = pb.ProgressBar(maxval=60, \
                    widgets=[pb.ETA(), pb.Bar('=', '[', ']'), ' ', pb.Percentage()])

    logger.info("Time integration")
    times = np.linspace(0, 3.0e-10, 61)
    #times = np.linspace(0, 3.0e-10, 100000)
    for counter, t in enumerate(times):
        bar.update(counter)

        # Integrate
        sim.run_until(t)
        print counter
        print ("press return to continue")
        _ = raw_input()

        # Save averages to file
        mx, my, mz = sim.m_average
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
            np.save(os.path.join(MODULE_DIR, "finmag%s_exch_density.npy"%demagsolver), np.array(finmag_exch))
            np.save(os.path.join(MODULE_DIR, "finmag%s_demag_density.npy"%demagsolver), np.array(finmag_demag))
    fh.close()
    fe.close()

def test_compare_averages():
    for demagsolver in demagsolvers:     
        if not os.path.isfile(os.path.join(MODULE_DIR, demagsolver+"averages.txt")) \
           or (os.path.getctime(os.path.join(MODULE_DIR, demagsolver+"averages.txt")) <
               os.path.getctime(os.path.abspath(__file__))):
            run_finmag(demagsolver)

        computed = np.loadtxt(os.path.join(MODULE_DIR, demagsolver+"averages.txt"))
        ref = np.loadtxt(os.path.join(MODULE_DIR, demagsolver+"averages_ref.txt"))
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
            print "finmag%s:\n"%demagsolver, computed1
        assert err < REL_TOLERANCE, "Relative error = %g" % err

        # Plot nmag data
        nmagt = list(ref[:,0])*3
        nmagy = list(ref[:,1]) + list(ref[:,2]) + list(ref[:,3])
        p.plot(nmagt, nmagy, 'o', mfc='w', label='nmag')

        # Plot finmag data seperate plots for each finmag demag solver
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
        p.title("Finmag%s vs Nmag"%demagsolver)
        p.legend(loc='center right')
        p.savefig(os.path.join(MODULE_DIR, demagsolver+"exchange_demag.png"))
        #p.show
        p.close()
        print "Comparison of development written to %sexchange_demag.png"%demagsolver
        
def test_compare_energies():
    #Dictionary for finmag exchange results
    exch = {}
    #Dictionary for finmag demag results
    demag = {}
    for demagsolver in demagsolvers:
        ref = np.loadtxt(os.path.join(MODULE_DIR, demagsolver+"energies_ref.txt"))
        if not os.path.isfile(os.path.join(MODULE_DIR, demagsolver+"energies.txt")) \
           or (os.path.getctime(os.path.join(MODULE_DIR, demagsolver+"energies.txt")) <
               os.path.getctime(os.path.abspath(__file__))):
            run_finmag(demagsolver)

        computed = np.loadtxt(os.path.join(MODULE_DIR, demagsolver+"energies.txt"))
        assert np.size(ref) == np.size(computed), "Compare number of energies."

        vol = mesh_volume(mesh)*unit_length**mesh.topology().dim()
        #30x30x100nm^3 = 30x30x100=9000

        # Compare exchange energy
        exch[demagsolver] = computed[:, 0]/vol
        exch_nmag = ref[:, 0]

        diff = abs(exch[demagsolver] - exch_nmag)
        rel_diff = np.abs(diff / max(exch[demagsolver]))
        print "Exchange energy Finmag %s Nmag , max relative error:"%demagsolver, max(rel_diff)
        assert max(rel_diff) < REL_TOLERANCE, \
                "Max relative error in Finmag %s Nmag exchange energy is %g"%(demagsolver, max(rel_diff))

        # Compare demag energy
        demag[demagsolver] = computed[:, 1]/vol
        demag_nmag = ref[:, 1]

        diff = abs(demag[demagsolver] - demag_nmag)
        rel_diff = np.abs(diff / max(demag[demagsolver]))
        print "Finmag %s Nmag Demag energy, max relative error: %g"%(demagsolver, max(rel_diff))
        # Don't really know why this is ten times higher than everyting else.
        assert max(rel_diff) < REL_TOLERANCE*10, \
                "Max relative error in Finmag %s Nmag demag energy is %g" %(demagsolver,max(rel_diff))

    # Plot both FK and GCR demags on the same plot.
    p.title("Finmag vs Nmag Exchange energy")
    p.plot(exch_nmag, 'o', mfc='w', label='Nmag')
    p.plot(exch["FK"], label='Finmag FK')
    p.plot(exch["GCR"], label='Finmag GCR')
    p.xlabel("Time step")
    p.ylabel("$\mathsf{E_{exch}}$")
    p.legend()
    p.savefig(os.path.join(MODULE_DIR, "exchange_energy.png"))
    p.close()
    
    p.plot(demag_nmag, 'o', mfc='w', label='Nmag')
    p.plot(demag, label='Finmag')
    p.xlabel("Time step")
    p.ylabel("$\mathsf{E_{demag}}$")
    p.legend()
    p.savefig(os.path.join(MODULE_DIR, "demag_energy.png"))
    #p.show()%
    p.close()
    print "Energy plots written to exchange_energy.png and demag_energy.png"
    
def test_compare_energy_density():
    """
    After ten time steps, compute the energy density through
    the center of the bar (seen from x and y) from z=0 to z=100,
    and compare the results with nmag and oomf.

    """
    R = range(100)
    finmag_exch = {}
    nmag_demag = {}
    for demagsolver in demagsolvers:
        # Run simulation only if not run before or changed since last time.
        if not (os.path.isfile(os.path.join(MODULE_DIR, demagsolver+"finmag_exch_density.npy"))):
            run_finmag(demagsolver)
        elif (os.path.getctime(os.path.join(MODULE_DIR, demagsolver+"finmag_exch_density.npy")) <
              os.path.getctime(os.path.abspath(__file__))):
            run_finmag(demagsolver)
        if not (os.path.isfile(os.path.join(MODULE_DIR, demagsolver+"finmag_demag_density.npy"))):
            run_finmag(demagsolver)
        elif (os.path.getctime(os.path.join(MODULE_DIR, demagsolver+"finmag_demag_density.npy")) <
              os.path.getctime(os.path.abspath(__file__))):
            run_finmag(demagsolver)

        # Read finmag data
        finmag_exch = np.load(os.path.join(MODULE_DIR, demagsolver+"finmag_exch_density.npy"))
        finmag_demag = np.load(os.path.join(MODULE_DIR, demagsolve+"finmag_demag_density.npy"))

        # Read nmag data
        nmag_exch = [float(i) for i in open(os.path.join(MODULE_DIR, "nmag_exch_Edensity.txt"), "r").read().split()]
        nmag_demag = [float(i) for i in open(os.path.join(MODULE_DIR, "nmag_demag_Edensity.txt"), "r").read().split()]

        # Compare with nmag
        nmag_exch[demagsolver] = np.array(nmag_exch)
        nmag_demag[demagsolver] = np.array(nmag_demag)
        rel_error_exch_nmag = np.abs(finmag_exch[demagsolver] - nmag_exch)/np.linalg.norm(nmag_exch)
        rel_error_demag_nmag = np.abs(finmag_demag[demagsolver] - nmag_demag)/np.linalg.norm(nmag_demag)
        print "Finmag %s Exchange energy density, max relative error from nmag: %g"%(demagsolver, max(rel_error_exch_nmag))
        print "Finmag %s Demag energy density, max relative error from nmag: %g"%(demagsolver, max(rel_error_demag_nmag))
        assert max(rel_error_exch_nmag) < 3e-2, \
            "Finmag %s Exchange energy density, max relative error from nmag is %g" %(demagsolver,max(rel_error_exch_nmag))
        assert max(rel_error_demag_nmag) < 1e-2, \
            "Finmag %s Demag energy density, max relative error from nmag is %g" %(demagsolver,max(rel_error_demag_nmag))
    
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
    p.plot(R, finmag_exch["FK"], 'o-',
           R,finmag_exch["GCR"], 'v-',
           R,nmag_exch, 'x-',
           oommf_coords, oommf_exch, "+-")
    
    p.xlabel("nm")
    p.title("Exchange energy density")
    p.legend(["Finmag FK","Finmag GCR", "Nmag", "oommf"], loc="upper center")
    p.savefig(os.path.join(MODULE_DIR, "exchange_density.png"))
    p.close()

    # Plot demag energy density
    p.plot(R, finmag_demag["FK"],
           R, finmag_demag["GCR"],
           'o-', R, nmag_demag,
           'x-', oommf_coords,
           oommf_demag, "+-")
    
    p.xlabel("nm")
    p.title("Demag energy density")
    p.legend(["Finmag FK","Finmag GCR", "Nmag", "oommf"], loc="upper center")
    p.savefig(os.path.join(MODULE_DIR, "demag_density.png"))
    #p.show()
    p.close()
    print "Energy density plots written to exchange_density.png and demag_density.png"
    
if __name__ == '__main__':
    test_compare_averages()
    test_compare_energies()
    test_compare_energy_density()
