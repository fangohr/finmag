"""One test. We integrate

du
-- = -2*u
dt


using (i) scipy.integrate.odeint as one normally does and 

(ii) within the dolfin framework.

The test here is whether we update the right function (u) in the
right-hand side when using dolfin.

For dolfin, we solve the ODE above on a mesh, where on every mesh
point we should(!) have exactly the same value (in each time
timestep).

It turns out that there is a slight deviation (within one timestep)
across the mesh. This, however, is of the order of 1-e16 (and thus the
usually numeric noise), and grows and shrinks over time.

While it is not clear where exactly this comes from (as the positional
coordinates do not enter the calculation?), this is not a blocker.

"""


import math
import logging
import numpy
import scipy.integrate
import dolfin as df
from dolfin import dx

#suppres dolfin outpet when solving matrix system
df.set_log_level(logging.WARNING)

def test_scipy_uprime_integration_with_fenics():

    NB_OF_CELLS_X = NB_OF_CELLS_Y = 2
    mesh = df.UnitSquare(NB_OF_CELLS_X, NB_OF_CELLS_Y)
    V = df.FunctionSpace(mesh, 'CG', 1)

    f = df.Expression("cos(t)",t=0)
    uprime = df.TrialFunction(V)
    uprev = df.interpolate(f,V)
    v = df.TestFunction(V)

    #ODE is du/dt= uprime = cos(t), exact solution is u(t)=sin(t)

    a = uprime*v*dx 
    L = f*v*dx

    uprime_solution = df.Function(V)
    uprime_problem  = df.LinearVariationalProblem(a, L, uprime_solution)
    uprime_solver   = df.LinearVariationalSolver(uprime_problem)

    def rhs_fenics(y,t):
        """A somewhat strange case where the right hand side is constant
        and thus we don't need to use the information in y."""
        #print "time: ",t
        uprev.vector()[:]=y
        f.t = t #dolfin needs to know the current time for cos(t)
        uprime_solver.solve()
        return uprime_solution.vector().array() 

    def rhs(y,t):
        """
        dy/dt = f(y,t) with y(0)=1
        dy/dt = -2y -> solution y(t) = c * exp(-2*t)"""
        return math.cos(t)

    T_MAX=2
    ts = numpy.arange(0,T_MAX+0.1,0.5)
    ysfenics=scipy.integrate.odeint(rhs_fenics, uprev.vector().array(), ts)

    def exact(t,y0=1):
        return y0*numpy.sin(t)

    print "With fenics:"
    err_abs = abs(ysfenics[-1][0]-exact(ts[-1])) #use value at mesh done 0 for check
    print "Error: abs=%g, rel=%g, y_exact=%g" % (err_abs,err_abs/exact(ts[-1]),exact(ts[-1]))
    fenics_error=err_abs


    print "Without fenics:"
    ys = scipy.integrate.odeint(rhs, 1, ts)

    err_abs = abs(ys[-1]-exact(ts[-1]))
    print "Error: abs=%g, rel=%g, y_exact=%g" % (err_abs,err_abs/exact(ts[-1]),exact(ts[-1]))

    non_fenics_error = float(err_abs)


    print("Difference between fenics and non-fenics calculation: %g" % abs(fenics_error-non_fenics_error))
    assert abs(fenics_error-non_fenics_error)<7e-15

    #should also check that solution is the same on all mesh points
    for i in range(ysfenics.shape[0]): #for all result rows
        #each row contains the data at all mesh points for one t in ts
        row = ysfenics[i,:]
        number_range = abs(row.min()-row.max())
        print "row: %d, time %f, range %g" % (i,ts[i],number_range)
        assert number_range < 16e-15

    #for debugging
    if False:
        from IPython.Shell import IPShellEmbed
        ipshell = IPShellEmbed()
        ipshell()

    if False:
            import pylab
            pylab.plot(ts,ys,'o')
            pylab.plot(ts,numpy.exp(-2*ts),'-')
            pylab.show()

if __name__== "__main__":
    test_scipy_uprime_integration_with_fenics()
