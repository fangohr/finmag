import logging 
import numpy as np
import dolfin as df
import finmag.sim.helpers as h
from test_exchange import ExchangeTmp

from finmag.native import llg as native_llg
from finmag.util.timings import timings

#default settings for logger 'finmag' set in __init__.py
#getting access to logger here
logger = logging.getLogger(name='finmag')

class LLB(object):
    """
    Solves the Baryakhtar or LLB equation.

    The equation reads

    .. math::

        \\frac{d\\vec{M}}{dt} = -\\gamma_{LL} (\\vec{M} \\times \\vec{H}) - \\alpha \\gamma_{LL} (\\vec{M} \\times [ \\vec{M} \\times \\vec{H}])

    where :math:`\\gamma_{LL} = \\frac{\\gamma}{1+\\alpha^2}`. In our code
    :math:`-\\gamma_{LL}` is referred to as *precession coefficient* and
    :math:`-\\alpha\\gamma_{LL}` as *damping coefficient*.

    """
    def __init__(self, S1, S3, do_precession=True):
        logger.debug("Creating LLG object.")
        timings.start('LLG-init')
        self.S1 = S1
        self.S3 = S3
        self.DG = df.FunctionSpace(S1.mesh(), "DG", 0)
        self._m = df.Function(self.S3)
        self._mu = df.Function(self.S3)
        self._Ms = df.Function(self.DG)
        self.count=1

        self.set_default_values()
        self.do_precession = do_precession
        timings.stop('LLG-init')
        self.Volume=None #will be computed on demand, and carries volume of the mesh

    def set_default_values(self):
        self._alpha_mult = df.Function(self.S1)
        self._alpha_mult.assign(df.Constant(1))
        self.alpha = 0.5 # alpha for solve: alpha * _alpha_mult

        self.gamma =  2.210173e5 # m/(As)
        #source for gamma:  OOMMF manual, and in Werner Scholz thesis, 
        #after (3.7), llg_gamma_G = m/(As).
        self.c = 1e11 # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self.Ms = 8.6e5 # A/m saturation magnetisation
        self.t = 0.0 # s
        self.pins = [] # nodes where the magnetisation gets pinned
        self._pre_rhs_callables=[]
        self._post_rhs_callables=[]
        self.interactions = []
    
    def set_pins(self, nodes):
        self._pins = np.array(nodes, dtype="int")
    def pins(self):
        return self._pins
    pins = property(pins, set_pins)

    def spatially_varying_alpha(self, baseline_alpha, multiplicator):
        """
        Accepts a dolfin function over llg.S1 of values
        with which to multiply the baseline alpha to get the spatially
        varying alpha.

        """
        self._alpha_mult = multiplicator
        self.alpha = baseline_alpha

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        # need to update the alpha vector as well, which is
        # why we have this property at all.
        self.alpha_vec = self._alpha * self._alpha_mult.vector().array()

    @property
    def M(self):
        """ the magnetisation, with length Ms """
        return self.Ms * self.m

    @property
    def M_average(self):
        """ the average magnetisation, computed with m_average() """
        return self.Ms * self.m_average

    @property
    def Ms(self):
        """ the magnetic moment """
        return self._Ms

    @Ms.setter
    def Ms(self, value):
        """
        Set the Ms
        """
        try:
            val = df.Constant(value)
        except:
            print 'Sorry, only a constant value is acceptable.'
            raise AttributeError

        tmp_Ms = df.interpolate(val, self.DG)

        self._Ms.vector()[:]=tmp_Ms.vector()
        print self.Ms

    @property
    def mu(self):
        """ the magnetic moment """
        return self._mu.vector().array()

    @mu.setter
    def mu(self, value):
        # Not enforcing unit length here, as that is better done
        # once at the initialisation of m.
        self._mu.vector()[:] = value


    
    def set_m(self, value, **kwargs):
        """
        Set the magnetisation (scaled automatically).
       
        There are several ways to use this function. Either you provide
        a 3-tuple of numbers, which will get cast to a dolfin.Constant, or
        a dolfin.Constant directly.
        Then a 3-tuple of strings (with keyword arguments if needed) that will
        get cast to a dolfin.Expression, or directly a dolfin.Expression.
        You can provide a numpy.ndarray of nodal values of shape (3*n,),
        where n is the number of nodes.
        Finally, you can pass a function (any callable object will do) which
        accepts the coordinates of the mesh as a numpy.ndarray of
        shape (3, n) and returns the magnetisation like that as well.

        You can call this method anytime during the simulation. However, when
        providing a numpy array during time integration, the use of
        the attribute m instead of this method is advised for performance
        reasons and because the attribute m doesn't normalise the vector.

        """
        if isinstance(value, tuple):
            if isinstance(value[0], str):
                # a tuple of strings is considered to be the ingredient
                # for a dolfin expression, whereas a tuple of numbers
                # would signify a constant
                val = df.Expression(value, **kwargs)
            else:
                val = df.Constant(value)
            new_m = df.interpolate(val, self.S3)
        elif isinstance(value, (df.Constant, df.Expression)):
            new_m = df.interpolate(value, self.S3)
        elif isinstance(value, (list, np.ndarray)):
            new_m = df.Function(self.S3)
            new_m.vector()[:] = value
        elif hasattr(value, '__call__'):
            coords = np.array(zip(* self.S3.mesh().coordinates()))
            new_m = df.Function(self.S3)
            new_m.vector()[:] = value(coords).flatten()
        else:
            raise AttributeError
        new_m.vector()[:] = h.fnormalise(new_m.vector().array())
        self._m.vector()[:] = new_m.vector()[:]

        tmp = df.assemble(self.Ms*df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx)
        self._mu.vector()[:]=tmp*self._m.vector()
        new_m.vector()[:]*=10
        self._mu.vector()[:]=new_m.vector()[:]



    def compute_effective_field(self):
        H_eff = np.zeros(self.mu.shape)
        for interaction in self.interactions:
            H_eff += interaction.compute_field()
        self.H_eff = H_eff
 
    def solve(self):
        for func in self._pre_rhs_callables:
            func(self)

        self.compute_effective_field()

        timings.start("LLG-compute-dmdt")
        # Use the same characteristic time as defined by c
        char_time = 0.1/self.c
        # Prepare the arrays in the correct shape
        m = self.mu
        m.shape = (3, -1)
        H_eff = self.H_eff
        H_eff.shape = (3, -1)
        H_eff_laplace = self.compute_laplace_effective_field()
        H_eff_laplace.shape = (3, -1)

        dMdt = np.zeros(m.shape)
        # Calculate dm/dt

        print self.alpha_vec,self.count
        self.do_precession=1
        self.count+=1
        native_llg.calc_baryakhtar_dmdt(m, H_eff,H_eff_laplace,
                                 self.t, dMdt, self.pins,
                                 self.gamma, self.alpha_vec, 0,
                                 char_time, self.do_precession)
        dMdt.shape = (-1,)

        timings.stop("LLG-compute-dmdt")

        for func in self._post_rhs_callables:
            func(self)

        return dMdt

    # Computes the dm/dt right hand side ODE term, as used by SUNDIALS CVODE
    def sundials_rhs(self, t, y, ydot):
        ydot[:] = self.solve_for(y, t)
        return 0

    def sundials_psetup(self, t, m, fy, jok, gamma, tmp1, tmp2, tmp3):
        if not jok:
            self.m = m
            self.compute_effective_field()
            self._reuse_jacobean = True

        return 0, not jok

    def sundials_psolve(self, t, y, fy, r, z, gamma, delta, lr, tmp):
        z[:] = r
        return 0

    # Computes the Jacobian-times-vector product, as used by SUNDIALS CVODE
    def sundials_jtimes(self, mp, J_mp, t, m, fy, tmp):
        timings.start("LLG-sundials-jtimes")

        assert m.shape == self.mu.shape
        assert mp.shape == m.shape
        assert tmp.shape == m.shape

        # First, compute the derivative H' = dH_eff/dt
        self.mu = mp
        Hp = tmp.view()
        Hp[:] = 0.
        
        for inter in self.interactions:
            if inter.in_jacobian:
                Hp[:] += inter.compute_field()

        if not hasattr(self, '_reuse_jacobean') or not self._reuse_jacobean:
        # If the field m has changed, recompute H_eff as well
            if not np.array_equal(self.mu, mu):
                self.mu = m
                self.compute_effective_field()

        m.shape = (3, -1)
        mp.shape = (3, -1)
        H = self.H_eff.view()
        H.shape = (3, -1)
        Hp.shape = (3, -1)
        J_mp.shape = (3, -1)
        # Use the same characteristic time as defined by c
        char_time = 0.1 / self.c
        native_llg.calc_baryakhtar_jtimes(m, H, mp, Hp, t, J_mp, self.gamma/(1+self.alpha**2),
                                   self.alpha, char_time, self.do_precession)
        # TODO: Store pins in a np.ndarray(dtype=int) and assign 0's in C++ code
        J_mp[:, self.pins] = 0.
        J_mp.shape = (-1, )
        m.shape = (-1,)
        mp.shape = (-1,)
        tmp.shape = (-1,)

        timings.stop("LLG-sundials-jtimes")

        # Nonnegative exit code indicates success
        return 0

    def solve_for(self, mu, t):
        self.mu = mu
        self.t = t
        value = self.solve() 
        return value 

    def compute_laplace_effective_field(self):
        grad_u = df.project(df.grad(self._mu))
        tmp=df.project(df.div(grad_u))
        return tmp.vector().array()
