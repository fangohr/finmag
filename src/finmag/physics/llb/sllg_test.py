import os
import dolfin as df
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from finmag.physics.llb.sllg import SLLG
from finmag.energies import Zeeman
from finmag.energies import Demag

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_random_number():
    from finmag.native.llb import RandomMT19937
    mt = RandomMT19937() 
    mt.initial_random(422353390)
    
    x=np.zeros(10000000,dtype=np.float)
    for i in range(100):    
        mt.gaussian_random_np(x)
        if i>80:
            plt.cla()
            plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)
            plt.grid(True)
            plt.savefig(os.path.join(MODULE_DIR, "test_mt19937_%d.png"%i))
        
        print 'step=',i
        

def plot_random_number_np():
    
    np.random.seed(422353390)
    
    for i in range(100):
        x=np.random.randn(10000000)    
    
        plt.cla()
        plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)
        plt.grid(True)
        plt.savefig(os.path.join(MODULE_DIR, "test_np_%d.png"%i))
    
    

if __name__ == "__main__":

    plot_random_number()
    plot_random_number_np()
    


