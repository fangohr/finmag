import dolfin as df
import numpy as np
from finmag.native import sundials
import finmag.native.llb as native_llb
from finmag.util.timings import timings


from finmag.energies import Zeeman
from finmag.energies import Demag
from finmag.llb.exchange import Exchange
from finmag.llb.anisotropy import LLBAnisotropy
from finmag.llb.material import Material


class LLB(object):
    def __init__(self, mat):
        self.material = mat
        self._m = mat._m
        self.m = self._m.vector().array()
        self.S1 = mat.V
        self.S3 = mat.S3
        
        self.dm_dt = np.zeros(self.m.shape)
        self.set_default_values()

    
    def set_default_values(self):

        self.gamma = 2.210173e5 # m/(As)
        self.alpha = 0.01
        self.t = 0.0 # s
        self.do_precession = True
        
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
        

    def compute_effective_field(self):
        H_eff = np.zeros(self.m.shape)
        for interaction in self.interactions:
            H_eff += interaction.compute_field()
        self.H_eff = H_eff        
    
    def sundials_rhs(self, t, y, ydot):
        self._m.vector().set_local(y)
        
        for func in self._pre_rhs_callables:
            func(self.t)

        self.compute_effective_field()

 
        timings.start("LLG-compute-dmdt")
        # Use the same characteristic time as defined by c

                
        native_llb.calc_llb_dmdt(self.m,
                                 self.H_eff,
                                 self.dm_dt,
                                 self.gamma,
                                 self.alpha,
                                 self.material.T,
                                 self.material.Tc,
                                 self.do_precession)


        timings.stop("LLG-compute-dmdt")

        for func in self._post_rhs_callables:
            func(self)
            
        ydot[:] = self.dm_dt[:]
        
        return 0
    
    
    def run_until(self, t):
        if t <= self.t:
            return
        
        self.integrator.advance_time(t, self.m)
        self._m.vector().set_local(self.m)




if __name__ == '__main__':
    x0 = y0 = z0 = 0
    x1 = 500e-9
    y1 = 10e-9
    z1 = 2e-9
    nx = 50
    ny = 1
    nz = 1
    mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz)
   
    mat = Material(mesh, name='FePt')
    mat.set_m((1, 0.2, 0))
    mat.T = 0
    
    llb = LLB(mat)
    llb.set_up_solver()
    
    
    
    app = Zeeman((1e3, 0, 0))
    app.setup(mat.S3, mat._m, Ms=mat.Ms0)
    llb.interactions.append(app)
    
    
    anis = LLBAnisotropy(mat.inv_chi_perp)
    anis.setup(mat.S3, mat._m, mat.Ms0)
    llb.interactions.append(anis)
    
    exch = Exchange(mat.A)
    exch.setup(mat.S3, mat._m, mat.Ms0, mat.m_e)
    llb.interactions.append(exch)
    
    demag = Demag("FK")
    demag.setup(mat.S3, mat._m, mat.Ms0)
    llb.interactions.append(demag)
    
    
    max_time = 1 * np.pi / (llb.gamma * 1e5)
    ts = np.linspace(0, max_time, num=200)

    mlist = []
    Ms_average = []
    for t in ts:
        print t
        llb.run_until(t)
        mlist.append(llb.m)
        df.plot(llb._m)
        
 
    
    df.interactive()
