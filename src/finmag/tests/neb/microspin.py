import os
import dolfin as df
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import colorConverter
from matplotlib.collections import PolyCollection, LineCollection


from finmag import Simulation as Sim
from finmag.energies import Exchange, DMI, UniaxialAnisotropy
from finmag.util.fileio import Tablereader

from finmag.sim.neb2 import NEB, plot_energy_3d, NEB_Sundials

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_simulation(mesh):
    
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9)
    
    sim.set_m((1,0,0))
    sim.add(UniaxialAnisotropy(-1e5, (0, 0, 1), name='Kp'))
    sim.add(UniaxialAnisotropy(1e4, (1, 0, 0), name='Kx'))
    
    return sim


def relax_system(sim):
    
    init_images=[(-1,0,0),(0,1,1),(1,0,0)]
    interpolations = [10,8]
    
    neb = NEB_Sundials(sim, init_images, interpolations, spring=5e5, name='neb')

    neb.relax(max_steps=500, save_ndt_steps=20, dt=1e-7)
    

def plot_data_2d():
    
    data = np.loadtxt('neb_energy.ndt')
    
    fig=plt.figure()
    xs = range(1, len(data[0,:]))
    plt.plot(xs, data[-1,1:], '.-')
    
    plt.grid()
    
    fig.savefig('last_energy.pdf')
        

if __name__ == "__main__":
    mesh = df.RectangleMesh(0,0,10,10,1,1)
    sim = create_simulation(mesh)
    relax_system(sim)

    
    plot_data_2d()
    plot_energy_3d('neb_energy.ndt')

