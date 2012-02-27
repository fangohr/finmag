import dolfin
import numpy
from scipy.integrate import odeint

from finmag.sim.llg import LLG

"""
The analytical solution of the LLG equation for a constant
applied field, based on Appendix B of Matteo's PhD thesis.

"""

def make_analytical_solution(H, alpha, gamma):
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

	def M(t):
	    return numpy.array([x(t), y(t), z(t)])

	return M

def test_macrospin_default_damping():
    """
    Compares the C/dolfin/odeint solution to the analytical one defined above.

    """
    
    x0 = y0 = z0 = 0
    x1 = y1 = z1 = 10e-9
    nx = ny = nz = 1
    mesh = dolfin.Box(x0, x1, y0, y1, z0, z1, nx, ny, nz)
    llg = LLG(mesh)
    llg.set_m0((1, 0, 0))
    llg.H_app = (0, 0, 1e5)

    EXCHANGE = False
    llg.setup(EXCHANGE)

    ts = numpy.linspace(0, 1e-9, num=100)
    ys = odeint(llg.solve_for, llg.m, ts)

    M_analytical = make_analytical_solution(1e5, llg.alpha, llg.gamma)

    TOLERANCE = 4e-2

    for i in range(len(ts)):
        
        M_computed = numpy.mean(ys[i].reshape((3, -1)), 1)
        print M_computed
        M_ref = M_analytical(ts[i])
        print M_ref
        diff_max = numpy.max(numpy.abs(M_computed - M_ref))

        assert diff_max < TOLERANCE, \
          "t=%e (i=%d) failed with diff=%e" % (ts[i],i,diff_max)

def test_macrospin_low_damping():
    """
    Compares the C/dolfin/odeint solution to the analytical one defined above.

    """
    
    x0 = y0 = z0 = 0
    x1 = y1 = z1 = 10e-9
    nx = ny = nz = 1
    mesh = dolfin.Box(x0, x1, y0, y1, z0, z1, nx, ny, nz)
    llg = LLG(mesh)
    llg.alpha = 0.02
    llg.set_m0((1, 0, 0))
    llg.H_app = (0, 0, 1e5)

    EXCHANGE = False
    llg.setup(EXCHANGE)

    ts = numpy.linspace(0, 1e-9, num=100)
    ys = odeint(llg.solve_for, llg.m, ts, rtol=1e-11)

    M_analytical = make_analytical_solution(1e5, llg.alpha, llg.gamma)

    TOLERANCE = 3e-3

    for i in range(len(ts)):
        
        M_computed = numpy.mean(ys[i].reshape((3, -1)), 1)
        M_ref = M_analytical(ts[i])
        diff_max = numpy.max(numpy.abs(M_computed - M_ref))

        assert diff_max < TOLERANCE, \
          "t=%e (i=%d) failed with diff=%e" % (ts[i],i,diff_max)
