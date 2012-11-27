import numpy as np
import dolfin as df
from finmag.util.consts import mu0, k_B
from finmag.util.meshes import mesh_volume

class RandomThermal(object):
    """
    Thermal field from Simone, 

    E. Martínez, L. López-Díaz, L. Torres and C.J. García-Cervera.
    Minimizing cell size dependence in micromagnetics simulations with thermal noise.
    J. Phys. D: Appl. Phys., vol. 40, pages 942-948, 2007. 

    and
    Phys. Rev. Lett. 90, 20 (2003) ???

    This field is not derived from an energy functional, and thus not 
    a real magnetic field. It's just a mathematical representation of
    the thermal noise in the context of the theory of stochastic
    processes and doesn't have any physical meaning, other than the
    correct statistical properties of the noise.

    """
    def __init__(self, alpha, gamma):
        """
        alpha could be a numpy array or a number

        """
        self.in_jacobian = False
        self.alpha = alpha
        self.gamma = gamma
        self.last_update_t = 0
        self.dt = 0

    def setup(self, S3, m, Ms, unit_length=1):
        mesh = S3.mesh()
        n_dim = mesh.topology().dim()
        self.V = mesh_volume(mesh=mesh) * unit_length**n_dim
        self.Ms = Ms
        self.output_shape = df.Function(S3).vector().array().shape

    def update(self, t, T):
        self.dt = t - self.last_update_t
        self.last_update_t = t
        self.T = T

    def compute_field(self):
        rnd = np.random.normal(loc=0.0, scale=1.0, shape=self.output_shape)
        amplitude = np.sqrt((10 * 2 * self.alpha * k_B * self.T) / (self.gamma * mu0 * self.Ms * self.V * self.dt))
        return amplitude * rnd

    def compute_energy(self):
        return 0 
