import dolfin as df
import numpy as np
import scipy.integrate.ode as scipy_ode
import matplotlib.pyplot as plt


import time

from finmag.drivers.llg_integrator import llg_integrator
from llb import LLB
from finmag.energies import Zeeman
from test_exchange import BaryakhtarExchange




def cross_times(a,b):
    assert(len(a)==len(b))
    assert(len(a)%3==0)

    res=np.zeros(len(a))
    for i in range(0,len(a),3):
        res[i]=a[i+1]*b[i+2]-a[i+2]*b[i+1]
        res[i+1]=a[i+2]*b[i]-a[i]*b[i+2]
        res[i+2]=a[i]*b[i+1]-a[i+1]*b[i]

    return res

def ode45_solve_llg():
    ns=[i for i in range(10)]
    Hz=1e6
    Happ=np.array([[0,0,Hz] for i in ns])
    Happ.shape=(-1,)
    print Happ

    Ms=8.6e2
    m0=np.array([1,2,3])
    M0=np.array([m0*Ms for i in ns])
    M0.shape=(-1,)
    gamma=2.21e5
    count=[]
    count_jac=[]
    def ode_rhs(t,M):
        tmp=cross_times(M,Happ)
        tmp[:]*=-gamma
        count.append(1)
        return tmp

    def jac(t,M):
        count_jac.append(1)
        B=np.zeros((len(M),len(M)))
        for i in range(0,len(M),3):
            B[i,i]=0
            B[i,i+1]=-Happ[i+2]
            B[i,i+2]=Happ[i+1]

            B[i+1,i]=Happ[i+2]
            B[i+1,i+1]=0
            B[i+1,i+2]=-Happ[i]

            B[i+2,i]=-Happ[i+1]
            B[i+2,i+1]=Happ[i]
            B[i+2,i+2]=0

        return B


    ode45=scipy_ode(ode_rhs)
    ode45.set_integrator('vode', method='bdf')

    ode45.set_initial_value(M0,0)



    max_time=2*np.pi/(gamma*Hz)*5
    dt=max_time/100
    ts=np.arange(0,max_time,dt)

    while ode45.successful() and ode45.t+dt<=ts[-1]:
        ode45.integrate(ode45.t+dt)
        print ode45.t,ode45.y,len(count),len(count_jac)


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

    llb = LLB(S1,S3,rtol=1e-6,atol=1e-10)
    llb.Ms=Ms
    llb.alpha = 0.0
    llb.set_m((1, 1, 1))
    H_app = Zeeman((0, 0, 1e5))
    H_app.setup(S3, llb._m,Ms=Ms)
    llb.interactions.append(H_app)
    exchange = BaryakhtarExchange(13.0e-12,1e-5)
    exchange.setup(S3,llb._m,llb._Ms)
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
    print llb.count
    save_plot(ts,Ms_average,'Ms_%g-time.png'%Ms)
    df.interactive()

def example1_sundials(Ms):
    x0 = y0 = z0 = 0
    x1 = y1 = z1 = 10
    nx = ny = nz = 1
    mesh = df.Box(x0, x1, y0, y1, z0, z1, nx, ny, nz)

    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1,dim=3)
    vis = df.Function(S3)

    llb = LLB(S1,S3)
    llb.alpha = 0.00
    llb.set_m((1, 1, 1))
    llb.Ms=Ms
    H_app = Zeeman((0, 0, 1e5))
    H_app.setup(S3, llb._m,Ms=Ms)
    llb.interactions.append(H_app)
    exchange = BaryakhtarExchange(13.0e-12,1e-2)
    exchange.setup(S3,llb._m,llb._Ms)
    llb.interactions.append(exchange)

    integrator = llg_integrator(llb, llb.M, abstol=1e-10, reltol=1e-6)

    max_time=2*np.pi/(llb.gamma*1e5)
    ts = np.linspace(0, max_time, num=50)

    mlist=[]
    Ms_average=[]
    for t in ts:
        integrator.advance_time(t)
        mlist.append(integrator.m.copy())
        llb.M=mlist[-1]
        vis.vector()[:]=mlist[-1]
        Ms_average.append(llb.M_average)
        df.plot(vis)
        time.sleep(0.0)
    print llb.count
    save_plot(ts,Ms_average,'Ms_%g-time-sundials.png'%Ms)
    df.interactive()

if __name__=="__main__":
    #ode45_solve_llg()
    example1(1)
    example1(8.6e5)
    example1_sundials(8.6e5)
    example1_sundials(1)
