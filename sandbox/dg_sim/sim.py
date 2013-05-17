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
from exchange import ExchangeDG as Exchange
#from finmag.energies import Demag

from fk_demag import FKDemagDG as Demag


logger = logging.getLogger(name='finmag')

class Sim(object):

    def __init__(self, mesh, Ms=8e5, unit_length=1.0, name='unnamed', auto_save_data=True):
        
        self.mesh = mesh
        self.unit_length=unit_length
        
        self.DG = df.FunctionSpace(mesh, "DG", 0)
        self.DG3 = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
        
        self._m = df.Function(self.DG3)
        
        self.Ms = Ms
        
        self.nxyz_cell=mesh.num_cells()
        self._alpha = np.zeros(self.nxyz_cell)
        self.m = np.zeros(3*self.nxyz_cell)
        self.H_eff = np.zeros(3*self.nxyz_cell)
        self.dm_dt = np.zeros(3*self.nxyz_cell)
        
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
        
        self.volumes = df.assemble(df.TestFunction(self.DG) * df.dx).array()
        
        self.real_volumes=self.volumes*self.unit_length**3
        self.Volume = np.sum(self.volumes)
        
        self.alpha = 0.5  # alpha for solve: alpha * _alpha_mult
        self.gamma = consts.gamma
        self.c = 1e11  # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
       
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
        

    def add(self,interaction):
        interaction.setup(self.DG3, 
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
        self._m.vector().set_local(self.m)

        self.t=t
        
        if self.auto_save_data:
            self.tablewriter.save()
        
    
    def sundials_rhs(self, t, y, ydot):
        self.t = t
        
        self.m[:]=y[:]
        self._m.vector().set_local(self.m)
        self.compute_effective_field()

        self.dm_dt[:]=0
        char_time = 0.1 / self.c
        self.m.shape=(3,-1)
        self.H_eff.shape=(3,-1)
        self.dm_dt.shape=(3,-1)
        
        
        native_llg.calc_llg_dmdt(self.m, self.H_eff, t, self.dm_dt, self.pins,
                                 self.gamma, self.alpha_vec,
                                 char_time, self.do_precession)

        self.m.shape=(-1)
        self.dm_dt.shape=(-1)
        self.H_eff.shape=(-1)
        
        ydot[:] = self.dm_dt[:]
        
        return 0
    
    def set_m(self,value):
        self.m[:]=helpers.vector_valued_dg_function(value, self.DG3, normalise=True).vector().array()
        self._m.vector().set_local(self.m)

    
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
    
    print mesh.num_cells(), mesh.num_vertices()
    
    demag=Demag()
    sim.add(demag)
    print demag.compute_field()
    
    
    for t in ts:
        sim.run_until(t)
        print sim.m_average