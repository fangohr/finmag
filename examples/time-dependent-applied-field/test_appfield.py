from scipy.integrate import ode
import numpy as np
import pylab
import dolfin as df
from finmag.sim.llg import LLG

def test_external_field_depends_on_t():

    simplices = 2
    L = 10e-9
    mesh = df.Interval(simplices, 0, L)
    omega=50*df.DOLFIN_PI/1e-9
    GHz=1e9
    omega= 100*GHz
    llg=LLG(mesh)
    llg.set_m0(df.Constant((1, 0, 0)))
    #This is the time dependent field
    H = df.Expression(("0.0", "0.0","H0*sin(omega*t)"), H0=1e5, omega=omega, t=0.0)

    #Add time dependent expression to LLG object
    llg._H_app_expression = H

    #define function that updates that expression, and the field 
    #object
    def update_H_ext(llg):
        print "update_H_ext being called for t=%g" % llg.t
        llg._H_app_expression.t = llg.t
        llg._H_app=df.interpolate(llg._H_app_expression,llg.V)

    #register this function to be called before (pre) the right hand side
    #of the ODE is evaluated
    llg._pre_rhs_callables.append(update_H_ext)

    #nothing special from here, just setting up time integration
    llg.setup()
    rhswrap = lambda t,y: llg.solve_for(y,t)
    r = ode(rhswrap).set_integrator('vode', method='bdf', with_jacobian=False)
    y0 = llg.m
    t0 = 0
    r.set_initial_value(y0, t0)

    tfinal = 0.3*1e-9
    dt = 0.001e-9

    #to gather data for later analysis
    mlist = []
    tlist = []

    #time loop
    while r.successful() and r.t < tfinal-dt:
        r.integrate(r.t + dt)
        print "Integrating time: %g" % r.t
        mlist.append(llg.m_average)
        tlist.append(r.t)

    #only plotting and data analysis from here on

    mx = [tmp[0] for tmp in mlist]
    my = [tmp[1] for tmp in mlist]
    mz = [tmp[2] for tmp in mlist]

    pylab.plot(tlist,mx,label='m_x')
    pylab.plot(tlist,my,label='m_y')
    pylab.plot(tlist,mz,label='m_z')
    pylab.xlabel('time [s]')
    pylab.legend()
    pylab.savefig('results.png')
    pylab.close()
    
    #Then try to fit sinusoidal curve through results
    def sinusoidalfit(t,omega,phi,A,B):
        return A*np.cos(omega*t+phi)+B

    #if scipy available
    try:
        import scipy.optimize
    except ImportError:
        print "Couldn't import scipy.optimize, skipping test"
    else:
        popt,pcov = scipy.optimize.curve_fit(
            sinusoidalfit,np.array(tlist),np.array(my),
            p0=(omega*1.04,0.,0.1,0.2))
            #p0 is the set of parameters with which the fitting 
            #routine starts 

        print "popt=",popt

        fittedomega,fittedphi,fittedA,fittedB=popt
        f=open("fittedresults.txt","w")

        print >>f, "Fitted omega           : %9g" % (fittedomega)
        print >>f, "Rel error in omega fit : %9g" % ((fittedomega-omega)/omega)
        print >>f, "Fitted phi             : %9f" % (fittedphi)
        print >>f, "Fitted Amplitude (A)   : %9f" % (fittedA)
        print >>f, "Fitted Amp-offset (B)  : %9f" % (fittedB)
        pylab.plot(tlist,my,label='my - simulated')
        pylab.plot(tlist,
                   sinusoidalfit(np.array(tlist),*popt),
                   '-x',label='m_y - fit')
        pylab.xlabel('time [s]')
        pylab.legend()
        pylab.savefig('fit.png')
        deviation = np.sqrt(sum((sinusoidalfit(np.array(tlist),*popt)-my)**2))/len(tlist)
        print >>f, "stddev=%g" % deviation
        f.close()

        assert (fittedomega-omega)/omega < 1e-4
        assert deviation < 5e-4


if __name__ == "__main__":
    test_external_field_depends_on_t()





