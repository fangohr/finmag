import dolfin as df
import instant
import os
import numpy as np
import finmag.sim.helpers as h
from finmag.sim.exchange import Exchange
from finmag.sim.anisotropy import UniaxialAnisotropy
from finmag.sim.dmi import DMI
from finmag.native import llg as native_llg
from finmag.demag.demag_solver import Demag

class LLG(object):

    def __init__(self, mesh, order=1, use_instant_llg=True, do_precession=True):
        self.mesh = mesh
        self.V = df.VectorFunctionSpace(self.mesh, 'Lagrange', order, dim=3)
        self.Volume = df.assemble(df.Constant(1)*df.dx, mesh=self.mesh)

        self.set_default_values()
        self.use_instant_llg = use_instant_llg
        self.do_precession = do_precession
        if use_instant_llg:
            self._solve = self.load_c_code()

    def set_default_values(self):
        self.alpha = 0.5
        self.gamma =  2.210173e5 # m/(As)
        #source for gamma:  OOMMF manual, and in Werner Scholz thesis, 
        #after (3.7), llg_gamma_G = m/(As).
        self.c = 1e11 # 1/s numerical scaling correction
                      # 0.1e12 1/s is the value used by default in nmag 0.2
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


    def load_c_code(self):
        """
        Loads the C-code in the file dmdt.c, that will later
        get called to compute the right-hand side of the LLG equation.

        """
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        with open(os.path.join(__location__, 'dmdt.c'), "r") as f:
            c_code = f.read()

        args = [["Mn", "M", "in"], ["Hn", "H", "in"],
                ["dMdtn", "dMdt", "out"], ["Pn", "P", "in"]]
        return instant.inline_with_numpy(c_code, arrays = args)
    
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
    
    def set_m0(self, value, **kwargs):
        """
        Set the initial magnetisation (scaled automatically).
        
        You can either provide a dolfin.Constant or a dolfin.Expression
        directly, or the ingredients for either, i.e. a tuple of numbers
        or a tuple of strings (with keyword arguments if needed), or provide
        the nodal values directly as a numpy array.
        
        """
        if isinstance(value, tuple):
            if isinstance(value[0], str):
                # a tuple of strings is considered to be the ingredient
                # for a dolfin expression, whereas a tuple of numbers
                # would signify a constant
                val = df.Expression(value, **kwargs)
            else:
                val = df.Constant(value)
            self._m0 = df.interpolate(val, self.V)
        elif isinstance(value, (df.Constant, df.Expression)):
            self._m0 = df.interpolate(value, self.V)
        elif isinstance(value, (list, np.ndarray)):
            self._m0 = df.Function(self.V)
            self._m0.vector()[:] = value
        else:
            raise AttributeError
        self._m0.vector()[:] = h.fnormalise(self._m0.vector().array())
        self._m.vector()[:] = self._m0.vector()[:]

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
        # - a (3d) value (for example using         self._H_app = df.interpolate(df.Constant(value), self.V))
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
            H_demag = self.demag.compute_field()
            self.H_eff += H_demag

    def solve(self):
        for func in self._pre_rhs_callables:
            func(self)

        self.compute_H_eff()

        if self.use_instant_llg:
            status, dMdt = self._solve(self.alpha, self.gamma, self.c,
                self.m, self.H_eff, self.m.shape[0], self.pins, self.do_precession)
        else:
            # Use the same characteristic time as defined by c
            char_time = 0.1/self.c
            # Prepare the arrays in the correct shape
            m = self.m
            m.shape = (3, -1)
            H_eff = self.H_eff
            H_eff.shape = (3, -1)
            dMdt = np.zeros(m.shape)
            # Calculate dm/dt
            native_llg.calc_llg_dmdt(m, H_eff, self.t, dMdt, self.gamma/(1.+self.alpha**2), self.alpha, char_time, self.do_precession)
            # TODO: Store pins in a np.ndarray(dtype=int) and assign 0's in C++ code
            dMdt[:, self.pins] = 0
            dMdt.shape = (-1,)
            status = 0

        for func in self._post_rhs_callables:
            func(self)

        if status == 0:
            return dMdt
        raise Exception("An error was encountered in the C-code; status=%d" % status)

    # Computes the Jacobean-times-vector product, as used by SUNDIALS CVODE
    def sundials_jtimes(self, mp, J_mp, t, m, fy, tmp):
        assert m.shape == self.m.shape
        assert mp.shape == m.shape
        assert tmp.shape == m.shape
        # First, compute the derivative of H
        Hp = tmp
        Hp[:] = 0.

        # Unfortunately we have to recompute H every time here
        self.m = m
        self.compute_H_eff()
        # Might be possible to avoid it later when we use a preconditioner, by computing it in pre_setup

        self.m = mp
        if self.use_exchange:
            Hp += self.exchange.compute_field()

        for ani in self._anisotropies:
            Hp += ani.compute_field()

        m.shape = (3, -1)
        mp.shape = (3, -1)
        H = self.H_eff.view()
        H.shape = (3, -1)
        Hp.shape = (3, -1)
        J_mp.shape = (3, -1)
        # Use the same characteristic time as defined by c
        char_time = 0.1 / self.c
        native_llg.calc_llg_jtimes(m, H, mp, Hp, t, J_mp, self.gamma/(1+self.alpha**2), self.alpha, char_time, self.do_precession)
        # TODO: Store pins in a np.ndarray(dtype=int) and assign 0's in C++ code
        J_mp[:, self.pins] = 0.
        J_mp.shape = (-1, )
        m.shape = (-1,)
        mp.shape = (-1,)
        tmp.shape = (-1,)

        # Nonnegative exit code indicates success
        return 0

    def solve_for(self, m, t):
        self.m = m
        self.t = t
        return self.solve()

    def add_uniaxial_anisotropy(self,K,a):
        self._anisotropies.append(UniaxialAnisotropy(self.V, self._m, K, a, self.Ms))

    def setup(self, use_exchange=True, use_dmi=False, use_demag=False,
              exchange_method="box-matrix-petsc",
              dmi_method="box-matrix-petsc",
              demag_method="GCR"):

        self.use_exchange = use_exchange
        if use_exchange:
            self.exchange = Exchange(self.V, self._m, self.A, 
                                     self.Ms, method=exchange_method)

        self.use_dmi = use_dmi
        if use_dmi:
            self.dmi = DMI(self.V, self._m, self.D, self.Ms, 
                           method = dmi_method)

        self.use_demag = use_demag
        if use_demag:
            self.demag = Demag(self.V, self._m, self.Ms, method=demag_method)
