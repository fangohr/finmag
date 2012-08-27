import logging 
import numpy as np
import dolfin as df
import finmag.util.helpers as h
import finmag.util.consts as consts

from finmag.native import llg as native_llg
from finmag.util.timings import timings

#default settings for logger 'finmag' set in __init__.py
#getting access to logger here
logger = logging.getLogger(name='finmag')

class LLG(object):
    """
    Solves the Landau-Lifshitz-Gilbert equation.

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
        self.set_default_values()
        self.do_precession = do_precession
        self.do_slonczewski = False
        timings.stop('LLG-init')
        self.Volume=None #will be computed on demand, and carries volume of the mesh

    def set_default_values(self):
        self._alpha_mult = df.Function(self.S1)
        self._alpha_mult.assign(df.Constant(1))
        self.alpha = 0.5 # alpha for solve: alpha * _alpha_mult

        self.gamma =  consts.gamma
        self.c = 1e11 # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self.Ms = 8.6e5 # A/m saturation magnetisation
        self.t = 0.0 # s
        self._pre_rhs_callables=[]
        self._post_rhs_callables=[]
        self._m = df.Function(self.S3)
        self._m.rename("m", "magnetisation") # gets displayed e.g. in Paraview when loading an exported VTK file
        self.pins = [] # nodes where the magnetisation gets pinned
        self.interactions = []
    
    def set_pins(self, nodes):
        """
        Hold the magnetisation constant for certain nodes in the mesh.

        Pass the indices of the pinned sites as *nodes*. Any type of sequence
        is fine, as long as the indices are between 0 (inclusive) and the highest index.
        This means you CANNOT use python style indexing with negative offsets counting
        backwards.

        """
        if len(nodes) > 0:
            nb_nodes_mesh = len(self._m.vector().array())/3
            if min(nodes) >= 0 and max(nodes) < nb_nodes_mesh:
                self._pins = np.array(nodes, dtype="int")
            else:
                logger.error("Indices of pinned nodes should be in [0, {}), were [{}, {}].".format(nb_nodes_mesh, min(nodes), max(nodes)))
        else:
            self._pins = np.array([], dtype="int")
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
        """The damping factor :math:`\\alpha`."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        # need to update the alpha vector as well, which is
        # why we have this property at all.
        self.alpha_vec = self._alpha * self._alpha_mult.vector().array()

    @property
    def M(self):
        """The magnetisation, with length Ms."""
        return self.Ms * self.m

    @property
    def M_average(self):
        """The average magnetisation, computed with m_average()."""
        return self.Ms * self.m_average

    @property
    def m(self):
        """The unit magnetisation."""
        return self._m.vector().array()

    @m.setter
    def m(self, value):
        # Not enforcing unit length here, as that is better done
        # once at the initialisation of m.
        self._m.vector()[:] = value

    @property
    def m_average(self):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`
        
        """
        #Compute volume if not done before
        if self.Volume == None:
            self.Volume = df.assemble(df.Constant(1)*df.dx, mesh=self._m.function_space().mesh())
        mx = df.assemble(df.dot(self._m, df.Constant([1,0,0])) * df.dx)
        my = df.assemble(df.dot(self._m, df.Constant([0,1,0])) * df.dx)
        mz = df.assemble(df.dot(self._m, df.Constant([0,0,1])) * df.dx)
        return np.array([mx, my, mz]) / self.Volume
    
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

    def compute_effective_field(self):
        H_eff = np.zeros(self.m.shape)
        for interaction in self.interactions:
            H_eff += interaction.compute_field()
        self.H_eff = H_eff
 
    def solve(self):
        for func in self._pre_rhs_callables:
            func(self.t)

        self.compute_effective_field()

        timings.start("LLG-compute-dmdt")
        # Use the same characteristic time as defined by c
        char_time = 0.1/self.c
        # Prepare the arrays in the correct shape
        m = self.m
        m.shape = (3, -1)
        H_eff = self.H_eff
        H_eff.shape = (3, -1)
        dMdt = np.zeros(m.shape)
        # Calculate dm/dt
        if self.do_slonczewski:
            native_llg.calc_llg_slonczewski_dmdt(
                m, H_eff, self.t, dMdt, self.pins,
                self.gamma, self.alpha_vec,
                char_time, 
                self.J, self.P, self.d, self.Ms, self.p)
        else:
            native_llg.calc_llg_dmdt(m, H_eff, self.t, dMdt, self.pins,
                                 self.gamma, self.alpha_vec, 
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
        # Note that ome of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        if not jok:
            self.m = m
            self.compute_effective_field()
            self._reuse_jacobean = True

        return 0, not jok

    def sundials_psolve(self, t, y, fy, r, z, gamma, delta, lr, tmp):
        # Note that ome of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        z[:] = r
        return 0

    # Computes the Jacobian-times-vector product, as used by SUNDIALS CVODE
    def sundials_jtimes(self, mp, J_mp, t, m, fy, tmp):
        timings.start("LLG-sundials-jtimes")

        assert m.shape == self.m.shape
        assert mp.shape == m.shape
        assert tmp.shape == m.shape

        # First, compute the derivative H' = dH_eff/dt
        self.m = mp
        Hp = tmp.view()
        Hp[:] = 0.
        
        for inter in self.interactions:
            if inter.in_jacobian:
                Hp[:] += inter.compute_field()

        if not hasattr(self, '_reuse_jacobean') or not self._reuse_jacobean:
        # If the field m has changed, recompute H_eff as well
            if not np.array_equal(self.m, m):
                self.m = m
                self.compute_effective_field()

        m.shape = (3, -1)
        mp.shape = (3, -1)
        H = self.H_eff.view()
        H.shape = (3, -1)
        Hp.shape = (3, -1)
        J_mp.shape = (3, -1)
        # Use the same characteristic time as defined by c
        char_time = 0.1 / self.c
        native_llg.calc_llg_jtimes(m, H, mp, Hp, t, J_mp, self.gamma,
                                   self.alpha_vec, char_time, self.do_precession, self.pins)
        J_mp.shape = (-1, )
        m.shape = (-1,)
        mp.shape = (-1,)
        tmp.shape = (-1,)

        timings.stop("LLG-sundials-jtimes")

        # Nonnegative exit code indicates success
        return 0

    def solve_for(self, m, t):
        self.m = m
        self.t = t
        value = self.solve() 
        return value

    def use_slonczewski(self, J, P, d, p):
        """
        Activates the computation of the Slonczewski spin-torque term in the LLG.

        J is the current density in A/m^2 as a dolfin expression,
        P is the polarisation (between 0 and 1),
        d the thickness of the free layer in m,
        p the direction (unit length) of the polarisation as a triple.

        """  
        self.do_slonczewski = True
        J = df.interpolate(J, self.S1)
        self.J = J.vector().array()
        assert P >= 0.0 and P <= 1.0
        self.P = P
        self.d = d
        polarisation = df.Function(self.S3)
        polarisation.assign(df.Constant((p)))
        self.p = polarisation.vector().array().reshape((3, -1))
