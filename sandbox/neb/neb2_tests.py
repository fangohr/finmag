import os
import dolfin as df
import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from neb2 import NEB_Sundials

    
class Sim3(object):
    """
    One spin with two anisotropies
    """
    def check(self,xs):
        x = xs[0]
        y = xs[1]
        if x > np.pi:
            xs[0] = np.pi
        elif x<0:
            xs[0]=0
            
        if y > np.pi:
            xs[1] -= 2*np.pi
        elif y < -np.pi:
            xs[1] += 2*np.pi
    
    def energy(self,xs):
        
        self.check(xs)
        
        x = xs[0]
        y = xs[1]
        
        
        return 1.0*np.sin(x)**2+2.0*np.sin(x)**2*np.sin(y)**2

    def gradient(self, xs):
        
        self.check(xs)
        
        x = xs[0]
        y = xs[1]
    
        gx = -2.0*np.sin(x)*np.cos(x)*(1+2*np.sin(y)**2)
        gy = -4.0*np.sin(y)*np.cos(y)*np.sin(x)**2
        
        return np.array([gx, gy])
    
def Sim3Test():
    
    init_images=[(0,0),(np.pi/2,np.pi/2-1),(np.pi,0)]
    interpolations = [12,9]
    
    sim = Sim3()
    
    neb = NEB_Sundials(sim, init_images, interpolations, name='neb', spring=0)
    
    neb.relax(max_steps=1000, stopping_dmdt=1e-6, dt=0.1)
    
    
if __name__ == '__main__':
    Sim3Test()


    

    
    

    