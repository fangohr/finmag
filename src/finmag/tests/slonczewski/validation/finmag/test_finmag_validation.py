import os
import numpy as np
from finmag.util.fileio import Tablereader
from run_validation import run_simulation

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
FINMAG_DYNAMICS_FILE = "finmag_validation_light.ndt"
NMAG_FILE = os.path.join(MODULE_DIR, "../nmag/averages_nmag5.txt")
OOMMF_FILE = os.path.join(MODULE_DIR, "../oommf/averages_oommf.txt")
EPSILON = 1e-16
TOLERANCE = 1e-4


def extract_magnetisation_dynamics():
    reader = Tablereader("finmag_validation.ndt")
    dynamics = np.array(reader["time", "m_x", "m_y", "m_z"]).T
    np.savetxt(FINMAG_DYNAMICS_FILE, dynamics, header="time mx my mz")


def test_against_nmag():
    run_simulation()

    if not os.path.isfile(FINMAG_DYNAMICS_FILE):
        extract_magnetisation_dynamics()
    finmag_dynamics = np.loadtxt(FINMAG_DYNAMICS_FILE)
    nmag_dynamics = np.loadtxt(NMAG_FILE)

    diff = np.abs(np.array(finmag_dynamics) - np.array(nmag_dynamics))
    print("Deviation is %s" % (np.max(diff)))
    assert np.max(diff[:, 0]) < EPSILON  # compare timesteps
    assert np.max(diff[:, 1:]) < TOLERANCE


def plot_dynamics():
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)

    if os.path.isfile(NMAG_FILE):
        nmag = np.loadtxt(NMAG_FILE)
        nmag[:, 0] *= 1e9
        ax.plot(nmag[:, 0], nmag[:, 1], "rx", label="nmag m_x")
        ax.plot(nmag[:, 0], nmag[:, 2], "bx", label="nmag m_y")
        ax.plot(nmag[:, 0], nmag[:, 3], "gx", label="nmag m_z")
    else:
        print "Missing nmag file."

    if os.path.isfile(OOMMF_FILE):
        oommf = np.loadtxt(OOMMF_FILE)
        oommf[:, 0] *= 1e9
        ax.plot(oommf[:, 0], oommf[:, 1], "r+", label="oommf m_x")
        ax.plot(oommf[:, 0], oommf[:, 2], "b+", label="oommf m_y")
        ax.plot(oommf[:, 0], oommf[:, 3], "g+", label="oommf m_z")
    else:
        print "Missing oommf file."

    finmag = np.loadtxt(FINMAG_DYNAMICS_FILE)
    finmag[:, 0] *= 1e9
    ax.plot(finmag[:, 0], finmag[:, 1], "k", label="finmag m_x")
    ax.plot(finmag[:, 0], finmag[:, 2], "k", label="finmag m_y")
    ax.plot(finmag[:, 0], finmag[:, 3], "k", label="finmag m_z")

    ax.set_xlim((0, 2))
    ax.set_xlabel("time (ns)")
    ax.legend()
    plt.savefig("dynamics.png")

if __name__ == "__main__":
    plot_dynamics()
