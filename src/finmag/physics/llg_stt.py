import dolfin as df
import numpy as np
import inspect
from aeon import default_timer
import finmag.util.consts as consts

from finmag.util import helpers
from finmag.physics.effective_field import EffectiveField
from finmag.util.meshes import nodal_volume
from finmag.native import llg as native_llg

import logging
log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


class LLG_STT(object):
    """
    Solves the Landau-Lifshitz-Gilbert equation with the nonlocal spin transfer torque.

    """
    
    def __init__(self, S1, S3, unit_length=1, average=False):
        self.S1 = S1
        self.S3 = S3
        self.unit_length = unit_length
        
        self.mesh = S1.mesh()
        
        self._m = df.Function(self.S3)
        # Arguments to _m.rename() below: (new_short_name, new_long_name).
        # These get displayed e.g. in Paraview when loading an
        # exported VTK file.
        self._m.rename("m", "magnetisation")
        
        self._delta_m = df.Function(self.S3)
        
        self.nxyz = len(self.m)
        self._alpha = np.zeros(self.nxyz/3)
        self.delta_m = np.zeros(self.nxyz)
        self.H_eff = np.zeros(self.nxyz)
        self.dy_m = np.zeros(2*self.nxyz) #magnetisation and delta_m
        self.dm_dt = np.zeros(2*self.nxyz) #magnetisation and delta_m
        
        self.set_default_values()
        self.effective_field = EffectiveField(self.S3, self._m, self.Ms, self.unit_length)

        self._t = 0
        
    def set_default_values(self):

        self.set_alpha(0.5)

        self.gamma = consts.gamma
        self.c = 1e11  # 1/s numerical scaling correction \
        #               0.1e12 1/s is the value used by default in nmag 0.2
        self.Ms = 8.6e5  # A/m saturation magnetisation

        self.vol = df.assemble(df.dot(df.TestFunction(self.S3),
                                      df.Constant([1, 1, 1])) * df.dx).array()
        self.real_vol = self.vol*self.unit_length**3


        self.pins=[]
        self._pre_rhs_callables = []
        self._post_rhs_callables = []
        self.interactions = []

    def set_parameters(self, J_profile=(1e10,0,0), P=0.5, D=2.5e-4, lambda_sf=5e-9, lambda_J=1e-9, speedup=1):
        
        self._J = helpers.vector_valued_function(J_profile, self.S3)
        self.J = self._J.vector().array()
        self.compute_gradient_matrix()
        self.H_gradm = df.PETScVector()
        
        self.P = P

        self.D = D/speedup
        self.lambda_sf = lambda_sf
        self.lambda_J = lambda_J
        
        self.tau_sf = lambda_sf**2/D*speedup
        self.tau_sd = lambda_J**2/D*speedup
        
        self.compute_laplace_matrix()
        self.H_laplace = df.PETScVector()
        
        self.nodal_volume_S3 = nodal_volume(self.S3)

    def set_pins(self, nodes):
        """
        Hold the magnetisation constant for certain nodes in the mesh.

        Pass the indices of the pinned sites as *nodes*. Any type of sequence
        is fine, as long as the indices are between 0 (inclusive) and the highest index.
        This means you CANNOT use python style indexing with negative offsets counting
        backwards.

        """
        if len(nodes) > 0:
            nb_nodes_mesh = len(self._m.vector().array()) / 3
            if min(nodes) >= 0 and max(nodes) < nb_nodes_mesh:
                self._pins = np.array(nodes, dtype="int")
            else:
                log.error("Indices of pinned nodes should be in [0, {}), were [{}, {}].".format(nb_nodes_mesh, min(nodes), max(nodes)))
        else:
            self._pins = np.array([], dtype="int")

    def pins(self):
        return self._pins
    pins = property(pins, set_pins)    
    
    @property
    def Ms(self):
        return self._Ms_dg

    @Ms.setter
    def Ms(self, value):
        #self._Ms_dg=helpers.scalar_valued_dg_function(value, self.S1)
        self._Ms_dg=helpers.scalar_valued_function(value, self.S1)
        self._Ms_dg.rename('Ms', 'Saturation magnetisation')
        self.volumes = df.assemble(df.TestFunction(self.S1) * df.dx)
        Ms = df.assemble(self._Ms_dg*df.TestFunction(self.S1)* df.dx).array()/self.volumes
        self._Ms = Ms.copy()
        self.Ms_av = np.average(self._Ms_dg.vector().array())

    @property
    def M(self):
        """The magnetisation, with length Ms."""
        #FIXME:error here
        m = self.m.view().reshape((3, -1))
        Ms = self.Ms.vector().array() if isinstance(self.Ms, df.Function) else self.Ms
        M = Ms * m
        return M.ravel()
    
    @property
    def M_average(self):
        """The average magnetisation, computed with m_average()."""
        volume_Ms = df.assemble(self._Ms_dg*df.dx,mesh=self.mesh)
        volume = df.assemble(self._Ms_dg*df.dx,mesh=self.mesh)
        return self.m_average*volume_Ms/volume

    @property
    def m(self):
        """The unit magnetisation."""
        return self._m.vector().array()

    @m.setter
    def m(self, value):
        # Not enforcing unit length here, as that is better done
        # once at the initialisation of m.
        self._m.vector()[:] = value
        self.dy_m.shape=(2,-1)
        self.dy_m[0][:]=value
        self.dy_m.shape=(-1,)
        
    @property
    def sundials_m(self):
        """The unit magnetisation."""
        return self.dy_m
    
    @sundials_m.setter
    def sundials_m(self, value):
        # used to copy back from sundials cvode
        self.dy_m[:] = value[:]
        self.dy_m.shape=(2,-1)
        self._m.vector()[:]=self.dy_m[0][:]
        self.dy_m.shape=(-1,)
        
    def m_average_fun(self,dx=df.dx):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`

        """

        mx = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([1, 0, 0])) * dx)
        my = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([0, 1, 0])) * dx)
        mz = df.assemble(self._Ms_dg*df.dot(self._m, df.Constant([0, 0, 1])) * dx)
        volume = df.assemble(self._Ms_dg*dx,mesh=self.mesh)

        return np.array([mx, my, mz]) / volume
    m_average=property(m_average_fun)
    
    def set_m(self, value, normalise=True, **kwargs):
        """
        Set the magnetisation (if `normalise` is True, it is automatically
        normalised to unit length).

        `value` can have any of the forms accepted by the function
        'finmag.util.helpers.vector_valued_function' (see its
        docstring for details).

        You can call this method anytime during the simulation. However, when
        providing a numpy array during time integration, the use of
        the attribute m instead of this method is advised for performance
        reasons and because the attribute m doesn't normalise the vector.

        """
        self.m = helpers.vector_valued_function(value, self.S3, normalise=normalise, **kwargs).vector().array()


    def set_alpha(self, value):
        """
        Set the damping constant :math:`\\alpha`.

        The parameter `value` can have any of the types accepted by the
        function :py:func:`finmag.util.helpers.scalar_valued_function` (see its
        docstring for details).

        """
        self._alpha[:]=helpers.scalar_valued_function(value,self.S1).vector().array()[:]
        


    def compute_gradient_matrix(self):
        """
        compute (J nabla) m , we hope we can use a matrix M such that M*m = (J nabla)m.

        """
        tau = df.TrialFunction(self.S3)
        sigma = df.TestFunction(self.S3)
        
        
        
        dim = self.S3.mesh().topology().dim()
        
        ty = tz = 0
        
        tx = self._J[0]*df.dot(df.grad(tau)[:,0],sigma)
        
        if dim >= 2:
            ty = self._J[1]*df.dot(df.grad(tau)[:,1],sigma)
        
        if dim >= 3:
            tz = self._J[2]*df.dot(df.grad(tau)[:,2],sigma)
        
        self.gradM = df.assemble(1/self.unit_length*(tx+ty+tz)*df.dx)
        
    def compute_gradient_field(self):

        self.gradM.mult(self._m.vector(), self.H_gradm)
        
        return self.H_gradm.array()/self.nodal_volume_S3
        
    def compute_laplace_matrix(self):
        
        u3 = df.TrialFunction(self.S3)
        v3 = df.TestFunction(self.S3)

        self.laplace_M = df.assemble(self.D/self.unit_length**2*df.inner(df.grad(u3),df.grad(v3))*df.dx)


    def compute_laplace_field(self):

        self.laplace_M.mult(self._delta_m.vector(), self.H_laplace)

        return -1.0*self.H_laplace.array()/self.nodal_volume_S3


    def sundials_rhs(self, t, y, ydot):
        self.t = t
        
        y.shape=(2,-1)
        self._m.vector().set_local(y[0])
        self._delta_m.vector().set_local(y[1])
        y.shape=(-1,)

        self.effective_field.update(t)
        H_eff = self.effective_field.H_eff  # alias (for readability)
        H_eff.shape = (3, -1)

        default_timer.start("sundials_rhs", self.__class__.__name__)
        # Use the same characteristic time as defined by c

        H_gradm = self.compute_gradient_field()
        H_gradm.shape=(3,-1)
        
        H_laplace = self.compute_laplace_field()
        H_laplace.shape = (3,-1)

        self.dm_dt.shape=(6,-1)
        
        m = self.m
        m.shape = (3, -1)
        
        char_time = 0.1 / self.c
        
        delta_m = self._delta_m.vector().array()
        delta_m.shape=(3,-1)
                
        native_llg.calc_llg_nonlocal_stt_dmdt(
                m, delta_m, H_eff, H_laplace, 
                H_gradm, 
                self.dm_dt, self.pins,
                self.gamma, self._alpha,
                char_time, self.P, 
                self.tau_sd, self.tau_sf, self._Ms)

        default_timer.stop("sundials_rhs", self.__class__.__name__)

        self.dm_dt.shape=(-1,)
        ydot[:] = self.dm_dt[:]
        
        H_gradm.shape = (-1,)
        H_eff.shape=(-1,)
        m.shape=(-1,)
        delta_m.shape=(-1,)

        return 0





if __name__ == '__main__':
    pass
