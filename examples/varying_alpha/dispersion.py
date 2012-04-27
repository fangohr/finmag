import os
from finmag.sim.llg import LLG
from finmag.sim.integrator import LLGIntegrator

import pytest
import pylab as p
import numpy as np
import dolfin as df
import progressbar as pb
import finmag.sim.helpers as h

import logging

logger = logging.getLogger(name='finmag')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 1e-4

def compute_dispersion(data,dx,dt,file_name):
    res=np.fft.fft2(data)
    res=np.fft.fftshift(res)
    res=np.abs(res)
    res=np.power(res,2)
    res=np.log10(res)
    m,n=res.shape
    print m,n
    file=open(file_name,'w')
    file.write('# kx (nm^-1)        frequency (GHz)        FFT_Power (arb. unit)\n')
    """Here we need double check ..."""
    for j in range(n):
        kx= (j+1-n/2.0)/(n*dx*1e9)*2*np.pi
        for i in range(m):
            f=(m/2.0-i)/(m*dt*1e9)
            file.write("%15g      %15g      %15g\n" % (kx, f, data[i][j]))
        file.write('\n')
    file.close()


def run_finmag():
    x_max = 300; y_max = 2; z_max = 2;
    mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 150, 1, 1)
    llg = LLG(mesh, mesh_units=1e-9)
    llg.Ms = 0.86e6
    llg.A = 13.0e-12
    llg.alpha = 0.01
    llg.set_m((1,0,0))

    GHz=1e9
    omega= 50*2*np.pi*GHz
    H = df.Expression(("0.0", "(x[0]<=2) ? ((t==0) ? H0: H0*sin(omega*t)/(omega*t)) : 0.0","0.0"), H0=1e5, omega=omega, t=0.0)
    def update_H_ext(llg):
        #print "update_H_ext being called for t=%g" % llg.t
        H.t = llg.t
        llg._H_app=df.interpolate(H,llg.V)
    llg._pre_rhs_callables.append(update_H_ext)
    
    llg.setup(use_exchange=True, use_dmi=False, use_demag=True, demag_method="FK")

    # Set up time integrator
    integrator = LLGIntegrator(llg, llg.m)

    dx = 2
    xs=[i*dx for i in xrange(0,x_max/dx)]

    data=[]
    tfinal = 1e-9
    dt = 0.001e-9
    
    times = np.linspace(0, tfinal, tfinal/dt + 1)
    for t in times:
        integrator.run_until(t)
        #update _m with values from integrator.m
        llg._m.vector()[:]=integrator.m[:] #or integrator.m
        print "Integrating time: %g" % t
        my=[llg._m(x,1,1)[1] for x in xs]
        data.append(my)

    #maybe we can use a TimeSeries object to store data 
    #(http://fenicsproject.org/documentation/dolfin/dev/python/programmers-reference/cpp/TimeSeries.html)
    

    return data,dx*1e-9,dt


if __name__ == '__main__':
    data,dx,dt=run_finmag()
    compute_dispersion(data,dx,dt,"dispersion.dat")
