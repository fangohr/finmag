import os
import dolfin as df
import numpy as np

import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import colorConverter
from matplotlib.collections import PolyCollection, LineCollection


from finmag import Simulation as Sim
from finmag.sim.neb import NEB
from finmag.energies import Exchange, DMI, UniaxialAnisotropy
from finmag.util.fileio import Tablereader

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def create_simulation(mesh):
    
    sim = Sim(mesh, Ms=8.6e5, unit_length=1e-9)
    
    sim.set_m((1,0,0))
    sim.add(UniaxialAnisotropy(-1e5, (0, 0, 1), name='Kp'))
    sim.add(UniaxialAnisotropy(1e4, (1, 0, 0), name='Kx'))
    
    return sim


def relax_system(sim):
    
    init_images=[(-1,0,0),(0,1,1),(1,0,0)]
    interpolations = [8,8]
    
    neb = NEB(sim, init_images, interpolations, name='neb')

    neb.relax(max_steps=500)
    
    #neb.relax(max_steps=100,dt=1e-7)


    
def plot_data():
    
    data = np.loadtxt('neb_energy.ndt')
    
    fig=plt.figure()
    ax = fig.gca(projection='3d')
    
    print data[1,1:]
    
    xs = range(1, len(data[0,:]))
    
    cc = lambda arg: colorConverter.to_rgba(arg, alpha=0.6)
    colors = [cc('r'), cc('g'), cc('b'),cc('y')]
    facecolors = []
    line_data = []
    for i in range(len(data[:,0])):
        line_data.append(list(zip(xs, data[i,1:]+1e-11)))
        facecolors.append(colors[i%4])
        
    poly = PolyCollection(line_data, facecolors=facecolors)
    poly.set_alpha(0.7)
    
    ax.add_collection3d(poly,zs=data[:,0], zdir='x')
    
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    ax.set_ylim3d(0, 21)
    ax.set_xlim3d(-1, max(data[:,0]))
    ax.set_zlim3d(0, 0.5e-11)
    
    fig.savefig('energy.pdf')

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
    plot_data()

