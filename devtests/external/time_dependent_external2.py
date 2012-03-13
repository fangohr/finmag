from scipy.integrate import ode
import numpy as np
import pylab
import dolfin as df
from finmag.sim.llg import LLG

#set_log_level(21)


simplices = 2
L = 10e-9
mesh = df.Interval(simplices, 0, L)
omega=50*df.DOLFIN_PI/1e-9

llg=LLG(mesh)

llg.set_m0(df.Constant((1, 0, 0)))

H = df.Expression(("0.0", "0.0","H0*sin(omega*t)"), H0=1e5, omega=omega, t=0.0)

#Need to modify llg object a little to have time dependent field
llg._H_app_expression = H

def update_H_ext(llg):
    print "update_H_ext being called for t=%g" % llg.t
    llg._H_app_expression.t = llg.t
    llg._H_app=df.interpolate(llg._H_app_expression,llg.V)

llg._pre_rhs_callables.append(update_H_ext)
llg.setup()

rhswrap = lambda t,y: llg.solve_for(y,t)
r = ode(rhswrap).set_integrator('vode', method='bdf', with_jacobian=False)

y0 = llg.m
t0 = 0
r.set_initial_value(y0, t0)

t1 = 0.3*1e-9
dt = 0.0025e-9

mlist = []
tlist = []

while r.successful() and r.t < t1-dt:
    r.integrate(r.t + dt)
    print "Integrating time: %g" % r.t
    if r.t > 0: #start gathering data after one ns (actually, also works from t=0)
        mlist.append(llg.m_average)
        tlist.append(r.t)

mx = [tmp[0] for tmp in mlist]
my = [tmp[1] for tmp in mlist]
mz = [tmp[2] for tmp in mlist]
print r.y
pylab.plot(tlist,mx,label='mx')
pylab.plot(tlist,my,label='my')
pylab.plot(tlist,mz,label='mz')
pylab.xlabel('time [s]')

def sinusoidalfit(t,omega,phi,A,B):
    return A*np.cos(omega*t+phi)+B

try:
    import scipy.optimize
except ImportError:
    print "Couldn't import scipy.optimize, skipping test"
else:
    popt,pcov = scipy.optimize.curve_fit(sinusoidalfit,np.array(tlist),np.array(my),p0=(omega*1.04,0.,0.1,0.2))
    print "popt=",popt
    
    fittedomega,fittedphi,fittedA,fittedB=popt

    print "Fitted omega         : %9f" % (fittedomega)
    print "Error in fitted omega: %9g" % ((fittedomega-omega)/omega)
    print "Fitted phi           : %9f" % (fittedphi)
    print "Fitted Amplitude     : %9f A/m" % (fittedA)
    print "Fitted Amp-offset    : %9f A/m" % (fittedB)
    pylab.plot(tlist,sinusoidalfit(np.array(tlist),*popt),'o:',label='fit')
    pylab.legend()

    deviation = np.sqrt(sum((sinusoidalfit(np.array(tlist),*popt)-my)**2))/len(tlist)
    print "stddev=%g" % deviation
    print "Written plot to file"
    pylab.savefig('time_dependent_external_field.pdf')
    assert (fittedomega-omega)/omega < 1e-4
    assert deviation < 5e-4





