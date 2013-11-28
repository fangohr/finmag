import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


from finmag import Simulation as Sim
from finmag.energies import Exchange, UniaxialAnisotropy

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_compute_dm():
    from finmag.sim.neb import compute_dm
    
    a1 = np.array([0,0,1])
    a2 = np.array([0,1,0])
    
    assert(compute_dm(a1,a2)**2-2.0)<1e-15


if __name__ == "__main__":
    test_compute_dm()


