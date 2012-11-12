import dolfin as df
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


import time

from baryakhtar import LLB
from finmag.energies import Zeeman
from finmag.energies import Demag
from exchange import Exchange
from finmag import Simulation as Sim
import finmag.energies as fe
#from finmag.energies import Exchange



def save_plot(x,y,filename):
    fig=plt.figure()
    plt.plot(x,y,'-.',markersize=3)
    fig.savefig(filename)

def example1(Ms=8.6e5):
    x0 = y0 = z0 = 0
    x1 = y1 = z1 = 10
    nx = ny = nz = 1
    mesh = df.Box(x0, x1, y0, y1, z0, z1, nx, ny, nz)
    
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1,dim=3)
    vis = df.Function(S3)

    llb = LLB(S1,S3)
    
    llb.alpha = 0.01
    llb.beta = 0.0
    llb.M0=Ms
    llb.set_M((Ms, 0, 0))
    llb.set_up_solver(jacobian=False)
    llb.chi=1e-4
    
    H_app = Zeeman((0, 0, 1e5))
    H_app.setup(S3, llb._M,Ms=1)
    llb.interactions.append(H_app)
    
    
    exchange = Exchange(13.0e-12,1e-2)
    exchange.chi=1e-4
    exchange.setup(S3,llb._M, Ms, unit_length=1e-9)
    
    llb.interactions.append(exchange)
    
    max_time=2*np.pi/(llb.gamma*1e5)
    ts = np.linspace(0, max_time, num=100)

    mlist=[]
    Ms_average=[]
    for t in ts:
        llb.run_until(t)
        mlist.append(llb.M)
        vis.vector()[:]=mlist[-1]
        Ms_average.append(llb.M_average)
        df.plot(vis)
        time.sleep(0.00)
    print 'llb times',llb.call_field_times
    save_plot(ts,Ms_average,'Ms_%g-time.png'%Ms)
    df.interactive()

def example2(Ms):
    x0 = y0 = z0 = 0
    x1 = 500
    y1 = 10
    z1 = 100
    nx = 50
    ny = 1
    nz = 1
    mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz)
    
    
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1,dim=3)
    
    llb = LLB(S1,S3)
    
    llb.alpha = 0.01
    llb.beta = 0.0
    llb.M0=Ms
    llb.set_M((Ms, 0, 0))
    llb.set_up_solver(jacobian=False)
    llb.chi=1e-4
    
 
    H_app = Zeeman((0, 0, 5e5))
    H_app.setup(S3, llb._M,Ms=1)
    llb.interactions.append(H_app)
    
    exchange = Exchange(13.0e-12,1e-2)
    exchange.chi=1e-4
    exchange.setup(S3,llb._M, Ms, unit_length=1e-9)
    llb.interactions.append(exchange)
    
    demag = Demag("FK")
    demag.setup(S3,llb._M, Ms=1)
    llb.interactions.append(demag)
    
    
    max_time = 1 * np.pi / (llb.gamma * 1e5)
    ts = np.linspace(0, max_time, num=100)


    for t in ts:
        print t
        llb.run_until(t)

        df.plot(llb._M)
        
    df.interactive()
    
    
def example3(Ms):
    x0 = y0 = z0 = 0
    x1 = 500
    y1 = 10
    z1 = 500
    nx = 50
    ny = 1
    nz = 1
    mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz)
    

    sim = Sim(mesh, Ms, unit_length=1e-9)
    sim.alpha = 0.01
    sim.set_m((1, 0, 0.1))
    
    H_app = Zeeman((0, 0, 5e5))
    sim.add(H_app)
    
    exch = fe.Exchange(13.0e-12)
    sim.add(exch)
    
    demag = Demag(solver="FK")
    sim.add(demag)
    

    llg=sim.llg
    max_time = 1 * np.pi / (llg.gamma * 1e5)
    ts = np.linspace(0, max_time, num=100)

    for t in ts:
        print t
        sim.run_until(t)

        df.plot(llg._m)
        
    df.interactive()

if __name__=="__main__":
    example2(8.6e5)
    
