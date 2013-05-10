import logging 
import numpy as np
import dolfin as df


from finmag.util import helpers
from finmag.util.meshes import mesh_volume

import finmag.util.consts as consts

from finmag.native import sundials
from finmag.native import llg as native_llg
from finmag.util.fileio import Tablewriter

from finmag.energies import Zeeman
from finmag.energies import Exchange
from finmag.energies import Demag




logger = logging.getLogger(name='finmag')

class Sim(object):

    def __init__(self, mesh, Ms=8e5, unit_length=1.0, name='unnamed', auto_save_data=True):
        
        self.mesh = mesh
        self.unit_length=unit_length
        
        self.S1 = df.FunctionSpace(mesh, "Lagrange", 1)
        self.DG = df.FunctionSpace(mesh, "DG", 0)
        self.S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1,dim=3)
        self.DG3 = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
        
        self._m = df.Function(self.S3)
        self._m_x = df.Function(self.DG)
        self._m_y = df.Function(self.DG)
        self._m_z = df.Function(self.DG)
        
        self.H_fun=df.Function(self.S3)
        
        self.Ms = Ms
        
        self.nxyz=mesh.num_vertices()
        self.m_cg=np.zeros(3*self.nxyz)
        
        self.nxyz_cell=mesh.num_cells()
        self._alpha = np.zeros(self.nxyz_cell)
        self.m=np.zeros(3*self.nxyz_cell)
        self.H_eff=np.zeros(3*self.nxyz)
        
        self.H_eff_cell=np.zeros(3*self.nxyz_cell)
        self.dm_dt=np.zeros(3*self.nxyz_cell)
        
        self.set_default_values()
        
        self.auto_save_data = auto_save_data
        self.sanitized_name = helpers.clean_filename(name)
        
        if self.auto_save_data:
            self.ndtfilename = self.sanitized_name + '.ndt'
            self.tablewriter = Tablewriter(self.ndtfilename, self, override=True)
                
    def set_default_values(self, reltol=1e-8, abstol=1e-8, nsteps=10000000):
        self._alpha_mult = df.Function(self.DG)
        self._alpha_mult.assign(df.Constant(1))
        
        self._pins = np.array([], dtype="int")
        
        self.volumes = df.assemble(df.TestFunction(self.S1) * df.dx).array()
        
        self.real_volumes=self.volumes*self.unit_length**3
        self.Volume = mesh_volume(self.mesh)
        
        self.alpha = 0.5  # alpha for solve: alpha * _alpha_mult
        self.gamma = consts.gamma
        self.c = 1e11  # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
       
        self._m = df.Function(self.S3)
        self.do_precession = True
                        
        self._pins = np.array([], dtype="int")
        self.interactions = []
        
        self.t = 0
        
    
    def set_up_solver(self, reltol=1e-8, abstol=1e-8, nsteps=10000):
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.m)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)
        
        self.integrator = integrator


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
    def pins(self):
        return self._pins

        
    def compute_effective_field(self):
        self.H_eff[:]=0 
        for interaction in self.interactions:
            self.H_eff += interaction.compute_field()
        
        self.H_fun.vector().set_local(self.H_eff)
        
        H_fun_dg = df.interpolate(self.H_fun, self.DG3)
        
        self.H_eff_cell[:] = H_fun_dg.vector().array()[:]

 
    def add(self,interaction):
        interaction.setup(self.S3, 
                          self._m, 
                          self.Ms,
                          unit_length=self.unit_length)
        self.interactions.append(interaction)
       
            
    def run_until(self, t):
        
        if t < self.t:
            return
        elif t == 0:
            if self.auto_save_data:
                self.tablewriter.save()
            return
        
        self.integrator.advance_time(t,self.m)
        self.compute_cg_m_from_dg()
        self.t=t
        
        if self.auto_save_data:
            self.tablewriter.save()
    
    def compute_cg_m_from_dg(self):
        self.m.shape=(3,-1)
        self._m_x.vector().set_local(self.m[0])
        self._m_y.vector().set_local(self.m[1])
        self._m_z.vector().set_local(self.m[2])
        self.m.shape=(-1,)
        
        mnx = df.assemble(self._m_x*df.TestFunction(self.S1) * df.dx).array()
        mny = df.assemble(self._m_y*df.TestFunction(self.S1) * df.dx).array()
        mnz = df.assemble(self._m_z*df.TestFunction(self.S1) * df.dx).array()
        
        mcg = self.m_cg
        mcg.shape=(3,-1)
        mcg[0][:]=mnx/self.volumes
        mcg[1][:]=mny/self.volumes
        mcg[2][:]=mnz/self.volumes
        mcg/=np.sqrt(mcg[0]**2+mcg[1]**2+mcg[2]**2)
        mcg.shape=(-1,)
        
        self._m.vector().set_local(mcg)
        
    
    def sundials_rhs(self, t, y, ydot):
        self.t = t
        
        self.m[:]=y[:]
        
        self.compute_cg_m_from_dg()
        
        self.compute_effective_field()

        self.dm_dt[:]=0
        char_time = 0.1 / self.c
        self.m.shape=(3,-1)
        self.H_eff_cell.shape=(3,-1)
        self.dm_dt.shape=(3,-1)
        
        
        native_llg.calc_llg_dmdt(self.m, self.H_eff_cell, t, self.dm_dt, self.pins,
                                 self.gamma, self.alpha_vec,
                                 char_time, self.do_precession)

        self.m.shape=(-1)
        self.dm_dt.shape=(-1)
        self.H_eff_cell.shape=(-1)
        
        ydot[:] = self.dm_dt[:]
        
        return 0
    
    def set_m(self,value):
        self._m.vector().set_local(helpers.vector_valued_function(value, self.S3, normalise=True).vector().array())
        self._m_dg = df.interpolate(self._m,self.DG3)
        self.m[:]=self._m_dg.vector().array()[:]
    
    @property
    def m_average(self):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`

        """ 
        
        mx = df.assemble(df.dot(self._m, df.Constant([1, 0, 0])) * df.dx)
        my = df.assemble(df.dot(self._m, df.Constant([0, 1, 0])) * df.dx)
        mz = df.assemble(df.dot(self._m, df.Constant([0, 0, 1])) * df.dx)
     
        return np.array([mx, my, mz])/self.Volume
    


if __name__ == "__main__":
    mesh = df.BoxMesh(0, 0, 0, 5, 5, 5, 1, 1, 1)
    sim = Sim(mesh, 8.6e5, unit_length=1e-9)
    sim.alpha = 0.01
    sim.set_m((1, 0, 0))
    sim.set_up_solver()
    ts = np.linspace(0, 1e-10, 11)
    
    
    H0 = 1e6
    sim.add(Zeeman((0, 0, H0)))
    
    exchange = Exchange(13.0e-12)
    sim.add(exchange)
    
    demag=Demag(solver='FK')
    sim.add(demag)
    
    
    for t in ts:
        sim.run_until(t)
        print sim.m_average