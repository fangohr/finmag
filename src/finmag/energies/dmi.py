import dolfin as df
import numpy as np
import logging
from finmag.util.timings import mtimed
from energy_base import EnergyBase
from finmag.util.consts import mu0
from finmag.util import helpers

logger=logging.getLogger('finmag')

def dmi_manual_curl(m, D, dim):
    """Input arguments:

       m a dolfin 3d-vector function on a 1d or 2d space,
       D is the DMI constant
       dim is the mesh dimension.

       Returns the form to compute the DMI energy:

         D * df.inner(M, df.curl(M)) * df.dx                      (1)

       However, curl(M) cannot be computed on a 2d and 1d mesh.

       Instead of equation (1), curl(m) can be computed as:

         curlx = dmzdy - dmydz
         curly = dmxdz - dmzdx
         curlz = dmydx - dmxdy

       and the scalar product with m is:

         E = D * (mx*curlx + my*curly + mz*curlz) * df.dx

       Derivatives along the z direction are set to be zero for both 1d and 2d mesh,
       and derivatives along the y direction are zero for 1d mesh since the physics 
       does not change as a function of these coordinates.
    """

    gradm = df.grad(m)

    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]
    if dim == 2:
        dmxdy = gradm[0, 1]
        dmydy = gradm[1, 1]
        dmzdy = gradm[2, 1]
    else:
        dmxdy = 0
        dmydy = 0
        dmzdy = 0
    dmxdz = 0
    dmydz = 0
    dmzdz = 0

    curlx = dmzdy - dmydz
    curly = dmxdz - dmzdx
    curlz = dmydx - dmxdy

    return D * (m[0] * curlx + m[1] * curly + m[2] * curlz)

class DMI(EnergyBase):
    """
    Compute the Dzyaloshinskii-Moriya interaction (DMI) field.

    .. math::

        E_{\\text{DMI}} = \\int_\\Omega D \\vec{M} \\cdot (\\nabla \\times \\vec{M})  dx

    

    *Arguments*

        D
            the DMI constant

        method
            See documentation of EnergyBase class for details.


    *Example of Usage*

        .. code-block:: python

            import dolfin as df
            from finmag.energies.dmi import DMI

            # Define a mesh representing a cube with edge length L
            L = 1e-8
            n = 5
            mesh = df.BoxMesh(0, L, 0, L, 0, L, n, n, n)

            S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
            D  = 5e-3 # J/m**2 DMI constant
            Ms = 0.8e6 # A/m magnetisation saturation
            m  = df.project(Constant((1, 0, 0)), S3) # Initial magnetisation

            dmi = DMI(D)
            dmi.setup(S3, m, Ms)

            # Print energy
            print dmi.compute_energy()

            # DMI field
            H_dmi = dmi.compute_field()

            # Using 'box-matrix-numpy' method (fastest for small matrices)
            dmi_np = DMI(D, method='box-matrix-numpy')
            dmi_np.setup(S3, m, Ms)
            H_dmi_np = dmi_np.compute_field()
    """

    def __init__(self, D, method="box-matrix-petsc", name='DMI'):
        self.D_waiting_for_mesh = D
        self.name = name
        super(DMI, self).__init__(method, in_jacobian=True)

    @mtimed
    def setup(self, S3, m, Ms, unit_length=1):
        self.DMI_factor = df.Constant(1.0 / unit_length)
        self.S3 = S3
        self.D = helpers.scalar_valued_dg_function(self.D_waiting_for_mesh, self.S3.mesh())
        self.D.rename('D', 'DMI_constant')
        self.D_av = np.average(self.D.vector().array())
        del(self.D_waiting_for_mesh)
        
        meshdim = S3.mesh().topology().dim()
        if meshdim == 1:
            E_integrand = dmi_manual_curl(m, self.DMI_factor * self.D, dim=1)
        elif meshdim == 2:
            E_integrand = dmi_manual_curl(m, self.DMI_factor * self.D, dim=2)
        elif meshdim == 3:
            E_integrand = self.DMI_factor * self.D * df.inner(m, df.curl(m))

        super(DMI, self).setup(E_integrand, S3, m, Ms, unit_length)
