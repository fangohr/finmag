import dolfin
import numpy
from scipy.integrate import odeint

from finmag.sim.llg import LLG

"""
The analytical solution of the LLG equation for a constant
applied field, based on Appendix B of Matteo's PhD thesis. 
#TODO
#Add page number and equation number

"""

def make_analytic_solution(H, alpha, gamma):
	"""
	Returns a function with computes the magnetisation vector
	as a function of time. Takes the following parameters:
		- Ms the saturation magnetisation in A/m
		- H the magnitude of the applied field
		- alpha has no dimension
		- gamma alias oommfs gamma_G in m/A*s
	
	"""

	p = float(gamma) / (1 + alpha**2)
	theta0 = numpy.pi / 2
	t0 = numpy.log(numpy.sin(theta0)/(1 + numpy.cos(theta0))) / (p * alpha * H)

	# Matteo uses spherical coordinates,
	# which have to be converted to cartesian coordinates.
	
	def phi(t):
	    return p * H * t
	def cos_theta(t):
	    return numpy.tanh(p * alpha * H * (t - t0))
	def sin_theta(t):
	    return 1 / (numpy.cosh(p * alpha * H * (t - t0)))

	def x(t):
	    return sin_theta(t) * numpy.cos(phi(t))
	def y(t):
	    return sin_theta(t) * numpy.sin(phi(t))
	def z(t):
	    return cos_theta(t)

	def m(t):
	    return numpy.array([x(t), y(t), z(t)])

	return m

def test_llg_macrospin_analytic(alpha=0.5,max_t=1e-9):
    """
    Compares the C/dolfin/odeint solution to the analytical one defined above.

    """

    print "running test_llg_macrospin_analytic with alpha=%g" % alpha

    
    #define 3d mesh
    x0 = y0 = z0 = 0
    x1 = y1 = z1 = 10e-9
    nx = ny = nz = 1
    mesh = dolfin.Box(x0, x1, y0, y1, z0, z1, nx, ny, nz)
    llg = LLG(mesh)
    llg.alpha = alpha
    llg.set_m0((1, 0, 0))
    llg.H_app = (0, 0, 1e6)

    EXCHANGE = False
    llg.setup(EXCHANGE)

    ts = numpy.linspace(0, max_t, num=100)
    tsfine = numpy.linspace(0, max_t, num=1000)
    ys = odeint(llg.solve_for, llg.m, ts)
    print ys.shape
    print llg.gamma

    m_analytical = make_analytic_solution(1e6, alpha, llg.gamma)

    TOLERANCE = 1e-6 #tolerance on Ubuntu 11.10, VM Hans, 25/02/2012

    for i in range(len(ts)):
        
        m = numpy.mean(ys[i].reshape((3, -1)), 1)
        #print m
        m_ref = m_analytical(ts[i])
        #print M_ref
        diff_max = numpy.max(numpy.abs(m - m_ref))
	print "diff_max (ts=%5g) = %g" % (ts[i],diff_max)
        assert diff_max < TOLERANCE, \
          "t=%e (i=%d) failed with diff=%e" % (ts[i],i,diff_max)

    ys3d = ys.reshape((len(ys),3,8)).mean(-1) 
    mx = ys3d[:,0]
    my = ys3d[:,1]
    mz = ys3d[:,2]
    print "mx.shape",mx.shape
    print "m_analytical.shape",m_analytical(ts).shape
    m_exact = m_analytical(tsfine)
    mx_exact = m_exact[0,:] 
    my_exact = m_exact[1,:] 
    mz_exact = m_exact[2,:] 
    #make plot
    import pylab
    pylab.plot(ts,mx,'o',label='mx')
    pylab.plot(ts,my,'x',label='my')
    pylab.plot(ts,mz,'^',label='mz')
    pylab.plot(tsfine,mx_exact,'-',label='mx (exact)')
    pylab.plot(tsfine,my_exact,'-',label='my (exact)')
    pylab.plot(tsfine,mz_exact,'-',label='mz (exact)')
    pylab.xlabel('t [s]')
    pylab.ylabel('m=M/Ms')
    pylab.title('Macro spin behaviour, alpha=%g' % alpha)
    pylab.grid()
    pylab.legend()
    pylab.savefig('alpha-%04.2f.png' % alpha)
    pylab.savefig('alpha-%04.2f.pdf' % alpha)
    #pylab.show()
    pylab.close()
    return ys 

def test_different_alphas():
	test_llg_macrospin_analytic(alpha=1)
	test_llg_macrospin_analytic(alpha=0.1)
	test_llg_macrospin_analytic(alpha=0.02)
	
if __name__=="__main__":
	ys = test_llg_macrospin_analytic(alpha=1.0,max_t=1e-10)
	ys = test_llg_macrospin_analytic(alpha=0.5,max_t=1e-10)
	ys = test_llg_macrospin_analytic(alpha=0.1,max_t=2e-10)
	ys = test_llg_macrospin_analytic(alpha=0.01,max_t=2e-9)
