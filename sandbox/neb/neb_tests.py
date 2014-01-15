import os
import dolfin as df
import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import inspect
from neb import NEB_Sundials
from finmag.sim.neb import plot_energy_3d

def plot_data_2d():
    
    data = np.loadtxt('neb_energy.ndt')
    
    fig=plt.figure()
    xs = range(1, len(data[0,:]))
    plt.plot(xs, data[-1,1:], '.-')
    
    K = 1e4
    a = 30
    b = 10
    c = 10
    v = 4.0/3*np.pi*a*b*c*1e-27
    expected_energy = K*v + np.min(data[-1,1:])
    
    line_x = [xs[0],xs[-1]]
    line_y = expected_energy*np.array([1,1])
    #plt.plot(line_x, line_y, '-',label='expected')
    
    plt.legend()
    plt.grid()
    
    fig.savefig('last_energy.pdf')

class Sim(object):
        
    def energy(self,xs):
        x = xs[0]
        y = xs[1]
        return np.sin(x)**2+2*np.sin(y)**2

    def gradient(self, xs):
        x = xs[0]
        y = xs[1]
        gx = -2*np.sin(x)*np.cos(x)
        gy = -4*np.sin(y)*np.cos(y)
        
        return np.array([gx, gy])
    
class Sim2(object):
        
    def energy(self,xs):
        x = xs[0]
        y = xs[2]
        return np.sin(x)**2+2*np.sin(y)**2

    def gradient(self, xs):
        x = xs[0]
        y = xs[2]
        gx = -2*np.sin(x)*np.cos(x)
        gy = -4*np.sin(y)*np.cos(y)
        
        return np.array([gx,0,gy,0])
    
    
class Sim3(object):
        
    def energy(self,xs):
        x = xs[0]
        y = xs[1]
        return np.sin(x)**2+2*np.sin(x)**2*np.sin(y)**2

    def gradient(self, xs):
        x = xs[0]
        y = xs[1]
        gx = -2*np.sin(x)*np.cos(x)*(1+2*np.sin(y)**2)
        gy = -4*np.sin(y)*np.cos(y)*np.sin(x)**2
        
        return np.array([gx, gy])
    
class SimTwoSpins(object):
        
    def energy(self,xs):
        x = xs[0]
        y = xs[3]
        return -(x**2 + 2*y**2)

    def gradient(self, xs):
        x = xs[0]
        y = xs[3]
        gx = 2*x
        gy = 4*y
        
        return np.array([gx,0,0, gy,0,0])
    

if __name__ == '__main__1':
    
    init_images=[(0,0),(np.pi,np.pi)]
    interpolations = [31]
    
    sim = Sim()
    
    neb = NEB_Sundials(sim, init_images, interpolations, name='neb', spring=0.5)
    
    neb.relax(max_steps=500, stopping_dmdt=1e-6)
    
    plot_data_2d()
    plot_energy_3d('neb_energy.ndt')
    
        
if __name__ == '__main__3':
    
    init_images=[(0,0,0,0),(np.pi,0,np.pi,0)]
    interpolations = [31]
    
    sim = Sim2()
    
    neb = NEB_Sundials(sim, init_images, interpolations, name='neb', spring=0.5)
    
    neb.relax(max_steps=500, stopping_dmdt=1e-6)
    
    plot_data_2d()
    plot_energy_3d('neb_energy.ndt')
    
if __name__ == '__main__':
    
    init_images=[(0,0),(np.pi/2,np.pi/2-0.1),(np.pi,0)]
    interpolations = [12,9]
    
    sim = Sim3()
    
    neb = NEB_Sundials(sim, init_images, interpolations, name='neb', spring=0.1)
    
    neb.relax(max_steps=1000, stopping_dmdt=1e-6)
    
    plot_data_2d()
    plot_energy_3d('neb_energy.ndt')
    
    
if __name__ == '__main__5':
    
    init_images=[(-1,0,0,-1,0,0),(1,0,0,1,0,0)]
    interpolations = [31]
    
    sim = SimTwoSpins()
    
    neb = NEB_Sundials(sim, init_images, interpolations, name='neb', spring=0.5, normalise=True)
    
    neb.relax(max_steps=500, stopping_dmdt=1e-6)
    
    plot_data_2d()
    plot_energy_3d('neb_energy.ndt')
    