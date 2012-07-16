import dolfin as df
import numpy as np
import matplotlib.pyplot as plt

from finmag.energies import Zeeman
from finmag.energies import Demag
from finmag.llb.exchange import Exchange
from finmag.llb.anisotropy import LLBAnisotropy
from finmag.llb.material import Material
from finmag.llb.llb import LLB


def average(m):
    m.shape=(3,-1)
    t=np.average(m, axis=1)
    m.shape=(-1)
    return np.sqrt(t[0]**2+t[1]**2+t[2]**2)

def saveplot(ts,me,filename):
    fig=plt.figure()
    plt.plot(ts,me)
    plt.xlabel('Time (ps)')
    plt.ylabel('me')
    fig.savefig(filename)

def SpinTest(mesh,T):
    mat = Material(mesh, name='FePt')
    mat.set_m((1, 1, 1))
    mat.T = 600
    
    llb = LLB(mat)
    llb.alpha=0.1
    llb.set_up_solver()
        
    llb.interactions.append(mat)
    
    max_time = 20e-12
    ts = np.linspace(0, max_time, num=101)

    
    me_average = []
    mx=[]
    mz=[]
    for t in ts:
        llb.run_until(t)
        me_average.append(average(llb.m))
        mx.append(llb.m[0])
        mz.append(llb.m[-1])
        
    saveplot(ts,me_average,'tt.png')
    saveplot(ts,mx,'mx.png')
    saveplot(ts,mz,'mz.png')

    return me_average




if __name__ == '__main__':
    x0 = y0 = z0 = 0
    x1 = 50e-9
    y1 = 10e-9
    z1 = 10e-9
    nx = 1
    ny = 1
    nz = 1
    mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz)
   
    print SpinTest(mesh,100)
    
    


