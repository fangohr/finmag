import os
import numpy as np
import pylab
import dolfin as df
from finmag.physics.llg import LLG
from finmag.energies import TimeZeeman
from finmag.integrators.llg_integrator import llg_integrator

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def test_external_field_depends_on_t():
    tfinal = 0.3*1e-9
    dt = 0.001e-9

    simplices = 2
    L = 10e-9
    mesh = df.IntervalMesh(simplices, 0, L)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    GHz=1e9
    omega= 100*GHz
    llg=LLG(S1, S3)
    llg.set_m(df.Constant((1, 0, 0)))

    #This is the time dependent field
    H_app_expr = df.Expression(("0.0", "0.0","H0*sin(omega*t)"), H0=1e5, omega=omega, t=0.0)
    H_app = TimeZeeman(H_app_expr)
    H_app.setup(S3, llg._m, Ms=8.6e5)
    #define function that updates that expression, and the field object
    def update_H_ext(t):
        print "update_H_ext being called for t=%g" % t
        H_app.update(t)
    llg.effective_field.add(H_app, with_time_update=update_H_ext)

    #nothing special from here, just setting up time integration
    integrator = llg_integrator(llg, llg.m)

    #to gather data for later analysis
    mlist = []
    tlist = []
    hext = []

    #time loop
    times = np.linspace(0, tfinal, tfinal/dt + 1)
    for t in times:
        integrator.advance_time(t)
        print "Integrating time: %g" % t
        mlist.append(llg.m_average)
        tlist.append(t)
        hext.append(H_app.H((0)))

    #only plotting and data analysis from here on

    mx = [tmp[0] for tmp in mlist]
    my = [tmp[1] for tmp in mlist]
    mz = [tmp[2] for tmp in mlist]

    pylab.plot(tlist,mx,label='m_x')
    pylab.plot(tlist,my,label='m_y')
    pylab.plot(tlist,mz,label='m_z')
    pylab.xlabel('time [s]')
    pylab.legend()
    pylab.savefig(os.path.join(MODULE_DIR, 'results.png'))
    pylab.close()

    #if max_step is not provided, or chosen too large,
    #the external field appears not smooth in this plot.
    #What seems to happen is that the ode integrator
    #returns the solution without calling the rhs side again
    #if we request very small time steps.
    #This is only for debugging.
    pylab.plot(tlist,hext,'-x')
    pylab.ylabel('external field [A/m]')
    pylab.xlabel('time [s]')
    pylab.savefig(os.path.join(MODULE_DIR, 'hext.png'))
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
        f=open(os.path.join(MODULE_DIR, "fittedresults.txt"),"w")

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
        pylab.savefig(os.path.join(MODULE_DIR, 'fit.png'))
        deviation = np.sqrt(sum((sinusoidalfit(np.array(tlist),*popt)-my)**2))/len(tlist)
        print >>f, "stddev=%g" % deviation
        f.close()

        assert (fittedomega-omega)/omega < 1e-4
        assert deviation < 5e-4


if __name__ == "__main__":
    test_external_field_depends_on_t()
