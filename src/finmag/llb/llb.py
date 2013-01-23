import dolfin as df
import numpy as np
from finmag.native import sundials
import finmag.native.llb as native_llb
from finmag.util.timings import timings


from finmag.energies import Zeeman
from finmag.energies import Demag
from finmag.llb.exchange import Exchange
from finmag.llb.material import Material


class LLB(object):
    def __init__(self, mat):
        self.material = mat
        self._m = mat._m
        self.m = self._m.vector().array()
        self.S1 = mat.V
        self.S3 = mat.S3
        
        self.dm_dt = np.zeros(self.m.shape)
        self.H_eff = np.zeros(self.m.shape)
        self.set_default_values()

    
    def set_default_values(self):

        self.alpha = self.material.alpha
        self.gamma_G = 2.21e5 # m/(As)
        self.gamma_LL = self.gamma_G/(1. + self.alpha**2)
        
        self.t = 0.0 # s
        self.do_precession = True
        
        self.vol = df.assemble(df.dot(df.TestFunction(self.S3),
                                      df.Constant([1, 1, 1])) * df.dx).array()
        self.vol *= self.material.unit_length**3
        #print 'vol=',self.vol
        
        self._pre_rhs_callables = []
        self._post_rhs_callables = []
        self.interactions = []
        
    def set_up_solver(self, reltol=1e-8, abstol=1e-8, nsteps=10000):
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.m)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)
        
        self.integrator = integrator
        
    def set_up_stochastic_solver(self, dt=1e-13,use_evans2012_noise=True,rk=False):
        self.dt = dt
        self.use_evans2012_noise = use_evans2012_noise
        
        M_pred=np.zeros(self.m.shape)
        if rk:
            integrator = native_llb.HeunStochasticIntegrator(
                                    self.m,
                                    M_pred,
                                    self.material.T,
                                    self.vol,
                                    self.dt,
                                    self.gamma_LL,
                                    self.alpha,
                                    self.material.Tc,
                                    self.material.Ms0,
                                    self.do_precession,
                                    self.use_evans2012_noise,
                                    self.stochastic_rhs)
        
        else:
            integrator = native_llb.RungeKuttaStochasticIntegrator(
                                    self.m,
                                    M_pred,
                                    self.material.T,
                                    self.vol,
                                    self.dt,
                                    self.gamma_LL,
                                    self.alpha,
                                    self.material.Tc,
                                    self.material.Ms0,
                                    self.do_precession,
                                    self.stochastic_rhs)
                                  
        self.integrator = integrator
        

    def compute_effective_field(self):
        self.H_eff[:]=0 
        for interaction in self.interactions:
            self.H_eff += interaction.compute_field()
        
    
    def stochastic_rhs(self, y):
        
        self._m.vector().set_local(y)
        
        for func in self._pre_rhs_callables:
            func(self.t)

        self.compute_effective_field()

        for func in self._post_rhs_callables:
            func(self)
            
    
    def sundials_rhs(self, t, y, ydot):
        self.t = t
        self._m.vector().set_local(y)
        
        for func in self._pre_rhs_callables:
            func(self.t)

        self.compute_effective_field()

 
        timings.start(self.__class__.__name__, "sundials_rhs")
        # Use the same characteristic time as defined by c
                
        native_llb.calc_llb_dmdt(self.m,
                                 self.H_eff,
                                 self.dm_dt,
                                 self.material.T,
                                 self.gamma_LL,
                                 self.alpha,
                                 self.material.Tc,
                                 self.do_precession)


        timings.stop(self.__class__.__name__, "sundials_rhs")

        for func in self._post_rhs_callables:
            func(self)
        
        ydot[:] = self.dm_dt[:]
        
        return 0
    
    
    def run_until(self, t):
        if t <= self.t:
            return
        
        self.integrator.advance_time(t, self.m)
        self._m.vector().set_local(self.m)
    
    def run_stochastic_until(self, t):
        if t <= self.t:
            return
        
        while self.t < t:
            #print self.t,self.m,self.H_eff
            self.integrator.run_step(self.H_eff)
            self._m.vector().set_local(self.m)
            self.t += self.dt
            
        


if __name__ == '__main__':
    x0 = y0 = z0 = 0
    x1 = 500
    y1 = 10
    z1 = 100
    nx = 50
    ny = 1
    nz = 1
    mesh = df.BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)
   
    mat = Material(mesh, name='FePt')
    mat.set_m((1, 0.2, 0))
    mat.T = 100
    mat.alpha=0.01
    
    llb = LLB(mat)
    llb.set_up_solver()
    
    llb.interactions.append(mat)
    
    
    app = Zeeman((0, 0, 5e5))
    app.setup(mat.S3, mat._m, Ms=mat.Ms0)
    llb.interactions.append(app)
        
    exch = Exchange(mat.A)
    exch.setup(mat.S3, mat._m, mat.Ms0, mat.m_e,unit_length=1e-9)
    llb.interactions.append(exch)
    
    demag = Demag("FK")
    demag.setup(mat.S3, mat._m, mat.Ms0)
    demag.demag.poisson_solver.parameters["relative_tolerance"] = 1e-10
    demag.demag.laplace_solver.parameters["relative_tolerance"] = 1e-10
    llb.interactions.append(demag)
    
    
    max_time = 1 * np.pi / (llb.gamma_LL * 1e5)
    ts = np.linspace(0, max_time, num=100)

    mlist = []
    Ms_average = []
    for t in ts:
        print t
        llb.run_until(t)
        mlist.append(llb.m)
        df.plot(llb._m)
        
 
    
    df.interactive()
