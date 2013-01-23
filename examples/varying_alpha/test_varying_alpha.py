import os
from finmag.sim.llg import LLG
from finmag.integrators.llg_integrator import llg_integrator
from finmag.energies import Exchange
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

import numpy as np
import dolfin as df

import logging

logger = logging.getLogger(name='finmag')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 1e-4

def save_plot(x,data,amp,k,phi,alpha,filename):
    fig=plt.figure()
    plt.plot(x,data,'.',markersize=3)
    my=model(x,amp,k,phi,alpha)
    plt.plot(x,my)
    fig.savefig(filename)  


def model(x,a,k,phi,alpha):
    """
    return the model of tea for its temperature decreases as the time flows
    """
    return a*np.cos(k*x+phi)*np.exp(-alpha*x)

def extract_parameters(x,data):
    """
    extract parameters from the ts and Ts according the model
    """
    res,res2=curve_fit(model,x,data,p0=(0.04, 0.1,0, 0.001))
    return tuple(res)


def run_nmag():
    x_max = 1000; y_max = 2; z_max = 2;
    mesh = df.BoxMesh(0, 0, 0, x_max, y_max, z_max, 500, 1, 1)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    llg = LLG(S1, S3, unit_length=1e-9)
    llg.set_m((1,0,0))
    llg.Ms = 0.86e6
    exchange = Exchange(13.0e-12)
    exchange.setup(S3, llg._m, llg.Ms)
    llg.alpha = 0.01

    GHz=1e9
    omega= 50*2*np.pi*GHz
    H = df.Expression(("0.0", "(x[0]<=2) ? ((t==0) ? H0: H0*sin(omega*t)) : 0.0","0.0"), H0=1e5, omega=omega, t=0.0)
    def update_H_ext(llg):
        print "update_H_ext being called for t=%g" % llg.t
        H.t = llg.t
        llg._H_app=df.interpolate(H,llg.V)
    llg._pre_rhs_callables.append(update_H_ext)
    
    # Set up time integrator
    integrator = llg_integrator(llg, llg.m)

    dx = 1
    xs=[i*dx for i in xrange(0,x_max/dx)]


    tfinal = 0.5e-9
    dt = 0.1e-9
    
    times = np.linspace(0, tfinal, tfinal/dt + 1)
    for t in times:
        integrator.run_until(t)
        print "Integrating time: %g" % t
        my=[llg._m(x,1,1)[1] for x in xs]
        

    return np.array(xs),np.array(my)


if __name__ == '__main__':
    xs,my=run_nmag()
    (amp,k,phi,alpha)=extract_parameters(xs,my)
    print k, alpha
    save_plot(xs,my,amp,k,phi,alpha,'res_detail.pdf')
