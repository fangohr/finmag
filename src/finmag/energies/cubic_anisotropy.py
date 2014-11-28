import logging
import numpy as np
import dolfin as df
from aeon import mtimed
from finmag.field import Field
from energy_base import EnergyBase
from finmag.util import helpers
from finmag.util.consts import mu0
from finmag.native import llg as native_llg

logger = logging.getLogger('finmag')


class CubicAnisotropy(EnergyBase):

    """
    Compute the cubic anisotropy field.

    *Arguments*
        K1, K2, K3
            The anisotropy constants.
        u1, u2, u3
            The anisotropy axes. Should be unit vectors.

    *Example of Usage*
            Refer to the UniaxialAnisotropy class.

    """

    def __init__(self, u1, u2, K1, K2=0, K3=0, name='CubicAnisotropy', assemble=False):
        """
        Define a cubic anisotropy with (fourth/six/eigth order) anisotropy
        constants `K1`, `K2`, `K3` (in J/m^3) and corresponding axes
        `u1`, `u2` and `u3`.

        if assemble = True, the box-assemble will be used, seems that box-assemble 
        method has introduced extra error!!!

        The constants and axes can be passed as df.Constant or df.Function or
        a general function, It is possible to specify spatially
        varying anisotropy by using df.Functions.

        """

        self.u1_value = u1
        self.u2_value = u2
        self.u3_value = np.cross(u1, u2)  # u3 perpendicular to u1 and u2

        self.K1_value = K1
        self.K2_value = K2
        self.K3_value = K3

        self.uv = 1.0 * np.array([self.u1_value, self.u2_value, self.u3_value])
        self.uv.shape = (-1,)

        self.name = name
        super(CubicAnisotropy, self).__init__("box-assemble", in_jacobian=True)

        self.assemble = assemble

    @mtimed
    def setup(self, m, Ms, unit_length=1):
        dofmap = m.functionspace.dofmap()
        S1 = df.FunctionSpace(
            m.mesh(), "Lagrange", 1, constrained_domain=dofmap.constrained_domain)

        S3 = df.VectorFunctionSpace(
            m.mesh(), "Lagrange", 1, 3, constrained_domain=dofmap.constrained_domain)

        self.K1_field = Field(S1, self.K1_value, name='K1')
        self.K2_field = Field(S1, self.K2_value, name='K2')
        self.K3_field = Field(S1, self.K3_value, name='K3')

        self.u1_field = Field(S3, self.u1_value, name='u1')
        self.u2_field = Field(S3, self.u2_value, name='u2')
        self.u3_field = Field(S3, self.u3_value, name='u3')
        
        self.volumes = df.assemble(df.TestFunction(S1) * df.dx)
        self.K1 = df.assemble(
            self.K1_field.f * df.TestFunction(S1) * df.dx).array() / self.volumes
        self.K2 = df.assemble(
            self.K2_field.f * df.TestFunction(S1) * df.dx).array() / self.volumes
        self.K3 = df.assemble(
            self.K3_field.f * df.TestFunction(S1) * df.dx).array() / self.volumes

        u1msq = df.dot(self.u1_field.f, m.f) ** 2
        u2msq = df.dot(self.u2_field.f, m.f) ** 2
        u3msq = df.dot(self.u3_field.f, m.f) ** 2

        E_term1 = self.K1_field.f * (u1msq * u2msq + u2msq * u3msq + u3msq * u1msq)
        E_term2 = self.K2_field.f * (u1msq * u2msq * u3msq)
        E_term3 = self.K3_field.f * \
            (u1msq ** 2 * u2msq ** 2 + u2msq **
             2 * u3msq ** 2 + u3msq ** 2 * u1msq ** 2)

        E_integrand = E_term1

        if self.K2_value != 0:
            E_integrand += E_term2

        if self.K3_value != 0:
            E_integrand += E_term3

        super(CubicAnisotropy, self).setup(E_integrand, m, Ms, unit_length)

        if not self.assemble:
            self.H = self.m.get_numpy_array_debug()
            self.Ms = self.Ms.get_numpy_array_debug()
            self.compute_field = self.__compute_field_directly

    def __compute_field_directly(self):

        m = self.m.get_numpy_array_debug()

        m.shape = (3, -1)
        self.H.shape = (3, -1)
        native_llg.compute_cubic_field(
            m, self.Ms, self.H, self.uv, self.K1, self.K2, self.K3)
        m.shape = (-1,)
        self.H.shape = (-1,)

        return self.H
