#GB what does llg mean?

import numpy as np
import logging 
import dolfin as df
import time
import finmag.sim.helpers as h
from finmag.sim.exchange import Exchange
from finmag.sim.anisotropy import UniaxialAnisotropy
from finmag.sim.dmi import DMI
from finmag.native import llg as native_llg
from finmag.demag.demag_solver import Demag # Deprecated, only for comparisson now
from finmag.demag.solver_fk_test import SimpleFKSolver
from finmag.util.timings import timings

#default settings for logger 'finmag' set in __init__.py
#getting access to logger here
logger = logging.getLogger(name='finmag')

class LLG(object):
    def __init__(self, mesh, order=1, unit_length=1, do_precession=True, called_from_sim=False):
        logger.info('Creating LLG object (rank=%s/%s) %s' % (df.MPI.process_number(),
                                                              df.MPI.num_processes(),
                                                              time.asctime()))
        if not called_from_sim:
            timings.reset()
        timings.start('LLG-init')
        
        self.mesh = mesh
        # save the units the mesh is expressed in,
        # so usually 1 (for m) for dolfin meshes, but 1e-9 (for nm) for netgen.
        self.unit_length = unit_length
        logger.debug("%s" % self.mesh)
    
        self.F = df.FunctionSpace(self.mesh, 'Lagrange', order)
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', order, dim=3)
        self.Volume = df.assemble(df.Constant(1)*df.dx, mesh=self.mesh)

        self.set_default_values()
        self.do_precession = do_precession
        timings.stop('LLG-init')

    def set_default_values(self):
        self._alpha_mult = df.Function(self.F)
        self._alpha_mult.assign(df.Constant(1))
        self.alpha = 0.5 # alpha for dmdt.c: alpha * _alpha_mult

        self.gamma =  2.210173e5 # m/(As)
        #source for gamma:  OOMMF manual, and in Werner Scholz thesis, 
        #after (3.7), llg_gamma_G = m/(As).
        self.c = 1e11 # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self.A = 1.3e-11 # J/m exchange constant
        self.D = 5e-3 # J/m DMI constant
        self.Ms = 8.6e5 # A/m saturation magnetisation
        self.t = 0 # s
        self.H_app = (0, 0, 0)
        self.H_dmi = (0, 0, 0) #DMI for Skyrmions
        self.pins = [] # nodes where the magnetisation gets pinned
        self._pre_rhs_callables=[]
        self._post_rhs_callables=[]
        self._anisotropies = []
        self._m = df.Function(self.V)
    
    def set_pins(self, nodes):
        self._pins = np.array(nodes, dtype="int")
    def pins(self):
        return self._pins
    pins = property(pins, set_pins)

    def spatially_varying_alpha(self, baseline_alpha, multiplicator):
        """
        Accepts a dolfin function over llg.F of values
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
    def m(self):
        """ the unit magnetisation """
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
        mx = df.assemble(df.dot(self._m, df.Constant([1,0,0])) * df.dx)
        my = df.assemble(df.dot(self._m, df.Constant([0,1,0])) * df.dx)
        mz = df.assemble(df.dot(self._m, df.Constant([0,0,1])) * df.dx)
        return np.array([mx, my, mz]) / self.Volume
    
    def set_m(self, value, **kwargs):
        """
        Set the magnetisation (scaled automatically).
        
        You can either provide a dolfin.Constant or a dolfin.Expression
        directly, or the ingredients for either, i.e. a tuple of numbers
        or a tuple of strings (with keyword arguments if needed), or provide
        the nodal values directly as a numpy array.

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
            new_m = df.interpolate(val, self.V)
        elif isinstance(value, (df.Constant, df.Expression)):
            new_m = df.interpolate(value, self.V)
        elif isinstance(value, (list, np.ndarray)):
            new_m = df.Function(self.V)
            new_m.vector()[:] = value
        else:
            raise AttributeError
        new_m.vector()[:] = h.fnormalise(new_m.vector().array())
        self._m.vector()[:] = new_m.vector()[:]

    def update_H_eff(self):
        raise NotImplementedError,"There should be no need to call this function"
 
    @property
    def H_app(self):
        return self._H_app.vector().array()

    @H_app.setter
    def H_app(self, value):
        #This needs reworking: need to be more flexible in the way we can set
        #H_app. Need similar flexibility as we have for providing m0. In particular,
        #we would like to set H_app from
        # - a (3d) value (for example using         
        #   self._H_app = df.interpolate(df.Constant(value), self.V))
        # - a dolfin expression 
        # (- a numpy vector that provides values for self._H_app.vector())
        
        self._H_app = df.interpolate(df.Constant(value), self.V)

    def compute_H_eff(self):
        #compute the effective field
        self.H_eff = self.H_app #can we avoid this if we don't use H_app?
        #self.H_eff *= 0.0 #set to zero
        if self.use_exchange:
            self.H_ex = self.exchange.compute_field()
            self.H_eff += self.H_ex
        if self.use_dmi:
            self.H_dmi = self.dmi.compute_field()
            self.H_eff += self.H_dmi
        for ani in self._anisotropies:
            H_ani = ani.compute_field()
            self.H_eff += H_ani
        if self.use_demag:
            self.H_demag = self.demag.compute_field()
            self.H_eff += self.H_demag

    def solve(self):
        for func in self._pre_rhs_callables:
            func(self)

        self.compute_H_eff()

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
        native_llg.calc_llg_dmdt(m, H_eff, self.t, dMdt, self.pins,
                                 self.gamma/(1.+self.alpha**2), self.alpha_vec, 
                                 char_time, self.do_precession)
        dMdt.shape = (-1,)

        timings.stop("LLG-compute-dmdt")

        for func in self._post_rhs_callables:
            func(self)

        return dMdt

    def compute_dmdt(self, m, H):
        """ Called from Simulation class. """
        timings.start("LLG-compute-dmdt")
        char_time = 0.1/self.c
        m.shape = (3, -1)
        H.shape = (3, -1)
        dmdt = np.zeros(m.shape) 
        native_llg.calc_llg_dmdt(m, H, self.t, dmdt, self.pins,
                                 self.gamma/(1.+self.alpha**2), self.alpha_vec, 
                                 char_time, self.do_precession)
        dmdt.shape = (-1, )
        timings.stop("LLG-compute-dmdt")
        return dmdt

    # Computes the dm/dt right hand side ODE term, as used by SUNDIALS CVODE
    def sundials_rhs(self, t, y, ydot):
        ydot[:] = self.solve_for(y, t)
        return 0

    def sundials_psetup(self, t, m, fy, jok, gamma, tmp1, tmp2, tmp3):
        if not jok:
            self.m = m
            self.compute_H_eff()
            self._reuse_jacobean = True

        return 0, not jok

    def sundials_psolve(self, t, y, fy, r, z, gamma, delta, lr, tmp):
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
        if self.use_exchange:
            Hp[:] = self.exchange.compute_field()
        else:
            Hp[:] = 0.

        for ani in self._anisotropies:
            Hp += ani.compute_field()

        if not hasattr(self, '_reuse_jacobean') or not self._reuse_jacobean:
        # If the field m has changed, recompute H_eff as well
            if not np.array_equal(self.m, m):
                self.m = m
                self.compute_H_eff()

        m.shape = (3, -1)
        mp.shape = (3, -1)
        H = self.H_eff.view()
        H.shape = (3, -1)
        Hp.shape = (3, -1)
        J_mp.shape = (3, -1)
        # Use the same characteristic time as defined by c
        char_time = 0.1 / self.c
        native_llg.calc_llg_jtimes(m, H, mp, Hp, t, J_mp, self.gamma/(1+self.alpha**2),
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

    def solve_for(self, m, t):
        self.m = m
        self.t = t
        value = self.solve() 
        return value

    def add_uniaxial_anisotropy(self,K,a):
        self._anisotropies.append(UniaxialAnisotropy(self.V, self._m, K, a, self.Ms))

    def setup(self, use_exchange=True, use_dmi=False, use_demag=False,
              exchange_method="box-matrix-petsc",
              dmi_method="box-matrix-petsc",
              demag_method="FK", demaglinsolTOL1 = 1e-7, demaglinsolTOL2 = 1e-7): 

        #add setup time to LLG-init
        timings.start('LLG-init')

        self.use_exchange = use_exchange
        if use_exchange:
            self.exchange = Exchange(self.V, self._m, self.A, 
              self.Ms, method=exchange_method, unit_length=self.unit_length)

        self.use_dmi = use_dmi
        if use_dmi:
            self.dmi = DMI(self.V, self._m, self.D, self.Ms, 
                           method = dmi_method, unit_length=self.unit_length)

        timings.stop('LLG-init')
        self.use_demag = use_demag
        # Using Weiwei's code as default
        if use_demag:   
            if demag_method == "weiwei":
                self.demag = SimpleFKSolver(self.V, self._m, self.Ms)

            elif demag_method == "FK":
                timings.start("Create demag problem")
                from finmag.demag.solver_fk import FemBemFKSolver
                timings.stop("Create demag problem")
                self.demag = FemBemFKSolver(self.mesh,self._m,Ms = self.Ms,
                                            unit_length=self.unit_length)

            elif demag_method == "GCR":
                timings.start("Create demag problem")
                from finmag.demag.solver_gcr import FemBemGCRSolver
                timings.stop("Create demag problem")
                self.demag = FemBemGCRSolver(self.mesh,self._m,Ms = self.Ms,
                                             unit_length=self.unit_length)

                
            else:
                self.demag = Demag(self.V, self._m, self.Ms, method=demag_method)
        

    def timings(self,n=20):
        """Prints an overview of wall time an number of calls for
        subparts of the code, listing up to n items, starting from 
        those that took the most wall time."""
        print timings.report_str(n)
