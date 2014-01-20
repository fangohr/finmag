import os
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sundials_ode import Test_Sundials
from sundials_ode import call_back

def plot_data(ts,ys):
    
    fig=plt.figure()
    
    plt.plot(ts, ys, '.-')
        
    #plt.legend()
    plt.grid()
    
    fig.savefig('ts_y.pdf')
    


def test1():
    y0 = np.array([0.1**i for i in range(10)])
    #y0 = np.array([1e-4,100,0,0,0,0,0,0,0,0])
    ts = Test_Sundials(call_back, y0)
    ts.run_step(50, min_dt=1e-5)
    
    plot_data(ts.ts,ts.ys)
    
    ts.print_info()


if __name__ == '__main__':
    
    test1()


    

    
    

    
