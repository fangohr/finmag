import numpy as np
import dolfin as df
from finmag import Simulation
from finmag.energies import Exchange, DMI, Demag, Zeeman
from finmag.energies.zeeman import TimeZeemanPython
import matplotlib.pyplot as plt

Rx = 5445
Ry = 60
Nx = 2178
Ny = 24

mesh = df.RectangleMesh(0,0,Rx,Ry,Nx,Ny)

def m_init_fun(pos):
    return np.random.random(3)-0.5

def m_init_skyrmion(pos):
    x = (pos[0])%45 - 22.5
    y = pos[1] - 30
     
    if x**2+y**2 < 5**2:
        return (0,0,-1)
    else:
        return (0,0,1)
    

class MyExpression(df.Expression):
    def __init__(self, h0, kc):
        self.h0 = h0
        self.kc = kc

    def eval(self, value, x):
        vy = 1.0
        
        for i in range(Ny):
            vy += np.sin(i*np.pi*x[1]/Ry)
        
        
        xp = self.kc * (x[0] - Rx/2.0)
        vy *= np.sinc(xp)
                
        value[0] = 0
        value[1] = vy*self.h0 
        value[2] = 0
        
    def value_shape(self):
        return (3,)
    

class AlphaExpression(df.Expression):
    def eval(self, value, x):
        if x[0] < 90:
            value[0]=(90-x[0])*100
        elif x[0] > 5355:
            value[0]=(x[0]-5355)*100
        else:
            value[0]=1.0

def plot_m(sim):
    df.plot(sim.llg._m)

def relax_system(mesh=mesh):
    
    Ms = 8.6e5
    sim = Simulation(mesh, Ms, pbc='1d', unit_length=1e-9, name = 'relax' )
    sim.set_m(m_init_skyrmion)

    A = 1.3e-11
    D = 4e-3
    sim.add(Exchange(A))
    sim.add(DMI(D))
    sim.add(Zeeman((0,0,0.4*Ms)))
    
    #sim.run_until(1e-9)
    #sim.schedule('save_vtk', at_end=True)
    #sim.schedule(plot_m, every=1e-10, at_end=True)
    
    sim.relax()
    df.plot(sim.llg._m)
    np.save('relaxed.npy',sim.m)
    #df.interactive()

def save_data(sim, xs):
    m = sim.llg._m
    my = np.array([m(x, 30)[1] for x in xs])
    np.save()


def find_skyrmion_center(fun):
    from scipy.signal import argrelmin
    
    xs = np.linspace(0, Rx, Nx+1)
    mzs = np.array([fun(x, 30)[2] for x in xs])
    mins = argrelmin(mzs)[0]

    all=[]
    xs_refine = np.linspace(-1.5, 1.5, 301)
    for i in range(len(mins)):
        mzs_refine = np.array([fun(x, 30)[2] for x in xs_refine+xs[mins[i]]])
        mins_fine = argrelmin(mzs_refine)[0]
        all.append(xs_refine[mins_fine[0]]+xs[mins[i]])
    
    
    #print all, len(all)
    for i in range(len(all)-1):
        print all[i+1]-all[i]
    
    
    xmin = all[0]
    xmax = all[-1]
    print xmin,xmax,len(mins), (xmax-xmin)/(len(mins)-1)
    
    return np.linspace(xmin, xmax, len(mins))
    #fig=plt.figure()


def excite_system():
    Ms = 8.6e5
    sim = Simulation(mesh, Ms,pbc='1d', unit_length=1e-9)
    
    sim.alpha = 0.0001
    sim.set_m(np.load('relaxed.npy'))
    
    alpha_expr = AlphaExpression()
    alpha_mult = df.interpolate(alpha_expr, sim.llg.S1)
    sim.spatial_alpha(0.0001, alpha_mult)
    
    #df.plot(alpha_mult)
    #df.interactive()
    #xs=find_skyrmion_center(sim.llg._m)
    #
    #assert(1==2)

    A = 1.3e-11
    D = 4e-3
    sim.add(Exchange(A))
    sim.add(DMI(D))
    sim.add(Zeeman((0,0,0.4*Ms)))
    
    GHz = 1e9
    omega = 50 * 2 * np.pi * GHz
    def time_fun(t):
        return np.sinc(omega*(t-50e-12))
    
    h0 = 1e3
    kc = 1.0/45.0
    H0 = MyExpression(h0,kc)
    
    sim.add(TimeZeemanPython(H0,time_fun))
    
    xs = find_skyrmion_center(sim.llg._m)
    ts =  np.linspace(0, 8e-9, 4001)
    
    np.save('xs.npy',xs)
    
    sim.create_integrator()
    sim.integrator.integrator.set_scalar_tolerances(1e-8, 1e-8)

    index = 0 
    for t in ts:
        
        sim.run_until(t)        
       
        np.save('data/m_%d.npy'%index, sim.llg.m)
        
        index += 1
   
    
    
if __name__ == '__main__':
    #relax_system()
    excite_system()

