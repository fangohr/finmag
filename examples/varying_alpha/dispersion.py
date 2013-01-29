import os
from finmag.sim.llg import LLG
from finmag.energies import Exchange
from finmag.integrators.llg_integrator import llg_integrator
import numpy as np
import dolfin as df
import logging

logger = logging.getLogger(name='finmag')

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
REL_TOLERANCE = 1e-4


def compute_dispersion(series,dx,file_name):
    times=series.vector_times()
    dt=times[1]-times[0]
    data=[]
    x=df.Vector()
    for t in times:
        series.retrieve(x, t, False)
        data.append(x.array())


    res=np.fft.fft2(data)
    res=np.fft.fftshift(res)
    res=np.abs(res)
    res=np.power(res,2)
    res=np.log10(res)
    m,n=res.shape
    print m,n
    
    freq=np.fft.fftfreq(m,d=dt*1e9)
    kx=np.fft.fftfreq(n,d=dx*1e9/(2*np.pi))
    freq=np.fft.fftshift(freq)
    kx=np.fft.fftshift(kx)

    f=open(file_name,"w")
    f.write('# kx (nm^-1)        frequency (GHz)        FFT_Power (arb. unit)\n')
    
    for j in range(n):
        for i in range(m):
            f.write("%15g      %15g      %15g\n" % (kx[n-j-1], freq[i], res[i][j]))
        f.write('\n')
    f.close()


def run_finmag():
    x_max = 2000; y_max = 2; z_max = 2;
    mesh = df.Box(0, 0, 0, x_max, y_max, z_max, 1000, 1, 1)
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
    H = df.Expression(("0.0", "(x[0]<=2) ? ((t==0) ? H0: H0*sin(omega*t)/(omega*t)) : 0.0","0.0"), H0=1e5, omega=omega, t=0.0)
    def update_H_ext(llg):
        #print "update_H_ext being called for t=%g" % llg.t
        H.t = llg.t
        llg._H_app=df.interpolate(H,llg.V)
    llg._pre_rhs_callables.append(update_H_ext)
    

    # Set up time integrator
    integrator = llg_integrator(llg, llg.m)

    dx = 2
    xs=[i*dx for i in xrange(0,x_max/dx)]

    series = df.TimeSeries("my")
    my=df.Vector()
    my.resize(len(xs));
    tfinal = 1e-9
    dt = 0.001e-9
    
    times = np.linspace(0, tfinal, tfinal/dt + 1)
    for t in times:
        integrator.advance_time(t)
        #update _m with values from integrator.m
        llg._m.vector()[:]=integrator.m[:] #or integrator.m
        print "Integrating time: %g" % t
        my[:]=np.array([llg._m(x,1,1)[1] for x in xs])
        series.store(my, t)

    #maybe we can use a TimeSeries object to store data 
    #(http://fenicsproject.org/documentation/dolfin/dev/python/programmers-reference/cpp/TimeSeries.html)
    

    return series,dx*1e-9


if __name__ == '__main__':
    first_run=0
    if first_run:
        series,dx=run_finmag()
        compute_dispersion(series,dx,"dispersion.dat")
    else:
        series = df.TimeSeries("my")
        dx=2e-9
        compute_dispersion(series,dx,"dispersion.dat")
