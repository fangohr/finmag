import logging 
import numpy as np
import dolfin as df


from finmag.util import helpers
import finmag.util.consts as consts

from finmag.native import sundials
from finmag.native import llg as native_llg
from finmag.util.timings import default_timer


#default settings for logger 'finmag' set in __init__.py
#getting access to logger here
logger = logging.getLogger(name='finmag')

class LLB(object):
    """
    Implementation of the Baryakhtar equation
    """
    def __init__(self, S1, S3,chi=0.001):
        self.S1 = S1
        self.S3 = S3
        self._M = df.Function(self.S3)
        self.M = self._M.vector().array()
        
        self.dm_dt = np.zeros(self.M.shape)
        self.H_eff = np.zeros(self.M.shape)
        
        self.call_field_times=0
        self.call_field_jtimes=0
        
        self.chi=chi
         
        self.set_default_values()
                
    def set_default_values(self):
        self._alpha_mult = df.Function(self.S1)
        self._alpha_mult.assign(df.Constant(1.0))
        self._beta_mult = df.Function(self.S1)
        self._beta_mult.assign(df.Constant(1.0))
        
        self.alpha = 0.5 # alpha for solve: alpha * _alpha_mult
        self.beta=0
        
        self.t = 0.0 # s
        self.do_precession = True
        
        u3 = df.TrialFunction(self.S3)
        v3 = df.TestFunction(self.S3)
        self.K = df.PETScMatrix()
        df.assemble(df.inner(df.grad(u3),df.grad(v3))*df.dx, tensor=self.K)
        self.H_laplace = df.PETScVector()
        
        self.vol = df.assemble(df.dot(df.TestFunction(self.S3), df.Constant([1, 1, 1])) * df.dx).array()
       
        self.gamma =  consts.gamma
        #source for gamma:  OOMMF manual, and in Werner Scholz thesis, 
        #after (3.7), llg_gamma_G = m/(As).
        self.c = 1e11 # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self.M0 = 8.6e5 # A/m saturation magnetisation
        self.t = 0.0 # s
        self._pins=np.zeros(self.S1.mesh().num_vertices(),dtype="int")
        self._pre_rhs_callables=[]
        self._post_rhs_callables=[]
        self.interactions = []
 

    @property
    def alpha(self):
        """The damping factor :math:`\\alpha`."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self.alpha_vec = self._alpha * self._alpha_mult.vector().array()
        
    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self.beta_vec = self._beta * self._beta_mult.vector().array()
    
    
    
    @property
    def pins(self):
        return self._pins
    
    @pins.setter
    def pins(self, value):
        self._pins[:]=helpers.scalar_valued_function(value,self.S1).vector().array()[:]
    
    def spatially_varying_alpha(self, baseline_alpha, multiplicator):
        """
        Accepts a dolfin function over llg.S1 of values
        with which to multiply the baseline alpha to get the spatially
        varying alpha.

        """
        self.alpha = baseline_alpha
        self._alpha_mult = multiplicator
        

    def set_M(self, value, **kwargs):
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
        self._M = helpers.vector_valued_function(value, self.S3, normalise=False)
        self.M[:]=self._M.vector().array()[:]
        

    def set_up_solver(self, reltol=1e-6, abstol=1e-6, nsteps=100000,jacobian=False):
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.M)
       
        if jacobian:
            integrator.set_linear_solver_sp_gmr(sundials.PREC_LEFT)
            integrator.set_spils_jac_times_vec_fn(self.sundials_jtimes)
            integrator.set_spils_preconditioner(self.sundials_psetup, self.sundials_psolve)
        else:
            integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
            
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)
        
        self.integrator = integrator


    def compute_effective_field(self):
        H_eff = np.zeros(self.M.shape)
        for interaction in self.interactions:
            H_eff += interaction.compute_field()
        
        #print 'M_',self._M.vector().array()
        #print 'heff=',H_eff
        self.H_eff = H_eff
        

 
    def compute_laplace_effective_field(self):
        #grad_u = df.project(df.grad(self._M))
        #tmp=df.project(df.div(grad_u))
        self.K.mult(self._M.vector(), self.H_laplace)
        return self.H_laplace.array()
       
            
    def run_until(self, t):
        
        if t <= self.t:
            return
        
        self.integrator.advance_time(t, self.M)
        self._M.vector().set_local(self.M)
        
    
    def sundials_rhs(self, t, y, ydot):
        self.t = t
        self._M.vector().set_local(y)
        
        for func in self._pre_rhs_callables:
            func(self.t)
        
        self.call_field_times+=1
        self.compute_effective_field()
        delta_Heff = self.compute_laplace_effective_field()
        #print 'delta_Heff',delta_Heff
 
        default_timer.start("sundials_rhs", self.__class__.__name__)
        # Use the same characteristic time as defined by c
        
        native_llg.calc_baryakhtar_dmdt(self._M.vector().array(), 
                                 self.H_eff,
                                 delta_Heff, 
                                 self.dm_dt,
                                 self.alpha_vec, 
                                 self.beta_vec,
                                 self.M0,
                                 self.gamma, 
                                 self.do_precession,
                                 self.pins)


        default_timer.stop("sundials_rhs", self.__class__.__name__)

        for func in self._post_rhs_callables:
            func(self)
        
        ydot[:] = self.dm_dt[:]
            
        return 0
    
    def sundials_jtimes(self, mp, J_mp, t, m, fy, tmp):
        """
       
        """

        default_timer.start("sundials_jtimes", self.__class__.__name__)
        self.call_field_jtimes+=1
    
        self._M.vector().set_local(m)
        self.compute_effective_field()
               
        print self.call_field_times,self.call_field_jtimes
        native_llg.calc_baryakhtar_jtimes(self._M.vector().array(),
                                   self.H_eff,
                                   mp, 
                                   J_mp, 
                                   self.gamma,
                                   self.chi,
                                   self.M0,
                                   self.do_precession,
                                   self.pins)
        
                            
        default_timer.stop("sundials_jtimes", self.__class__.__name__)
        
        self.sundials_rhs(t, m, fy)

        # Nonnegative exit code indicates success
        return 0

    def sundials_psetup(self, t, m, fy, jok, gamma, tmp1, tmp2, tmp3):
        # Note that some of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        return 0, not jok

    def sundials_psolve(self, t, y, fy, r, z, gamma, delta, lr, tmp):
        # Note that some of the arguments are deliberately ignored, but they
        # need to be present because the function must have the correct signature
        # when it is passed to set_spils_preconditioner() in the cvode class.
        z[:] = r
        return 0
    @property
    def M_average(self):
        """The average magnetisation, computed with m_average()."""

        tmp=self.M
        tmp.shape=(3,-1)
        res=np.average(tmp,axis=1)
        tmp.shape=(-1,)
        return res 
