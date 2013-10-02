import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


from finmag import Simulation as Sim
from finmag.energies import Exchange, Demag
from finmag.util.fileio import Tablereader

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def init_m(pos):
    x0 = 50
    y0 = 50
    r = 10
    
    x,y,z = pos
    fx,fy,fz = y0-y, x-x0, r

    R = np.sqrt(fx**2+fy**2+fz**2)

    return (fx,fy,fz)/R

def init_J(pos):
    
    return (1e12,0,0)


def relax_system(mesh):
    sim = Sim(mesh, Ms=8.0e5, unit_length=1e-9)
    sim.llg.gamma = 2.211e5
    
    sim.alpha = 1
    
    sim.add(Exchange(A=13e-12))
    sim.add(Demag())
    
    sim.set_m(init_m)
    
    sim.relax(stopping_dmdt=0.01)
    np.save('m0.npy',sim.m)
    df.plot(sim.llg._m)
    df.interactive()

def spin_length(sim):
    spin=sim.m
    spin.shape=(3,-1)
    length=np.sqrt(np.sum(spin**2,axis=0))
    spin.shape=(-1,)
    print sim.t,np.max(length),np.min(length)
    
def with_current(mesh):
    sim = Sim(mesh, Ms=8.0e5, unit_length=1e-9,name='stt')
    sim.llg.gamma = 2.211e5
    
    sim.alpha = 0.1
    sim.set_m(np.load('m0.npy'))
    
    sim.add(Exchange(A=13e-12))
    sim.add(Demag())
    
    sim.set_zhangli(init_J, 1.0,0.05)
    
    sim.schedule('save_ndt', every=1e-11)
    sim.schedule('save_vtk', every=5e-11)
    sim.schedule(spin_length, every=1e-11)
    
    sim.run_until(5e-9)
    df.plot(sim.llg._m)
    df.interactive()
    
def plot_data():
    data = Tablereader('stt.ndt')
    
    ts = data['time']
    
    fig=plt.figure()
    ms = 8.0e5/1e6
    plt.plot(ts*1e9,data['m_x']*ms,'.-',label='m_x')
    plt.plot(ts*1e9,data['m_y']*ms,'+-',label='m_y')
    plt.plot(ts*1e9,data['m_z']*ms,'-',label='m_z')
    plt.xlim([0,5])
    plt.ylim([-0.3,0.2])
    plt.xlabel('Time (ns)')
    plt.ylabel('Average Magnetisation ($10^6$A/m)')
    plt.gray()
    plt.legend()
    
    fig.savefig('averaged_m.pdf')
        

if __name__ == "__main__":
    mesh = df.BoxMesh(0, 0, 0, 100, 100, 10, 40, 40, 4)
    #relax_system(mesh)
    #with_current(mesh)
    plot_data()

