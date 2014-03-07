import dolfin as df
import numpy as np
import logging
from aeon import mtimed
from energy_base import EnergyBase
from finmag.util import helpers

logger = logging.getLogger('finmag')


def manual_times_curl(m, dim):
    """
    Returns m times curl of m on 1-dimensional and 2-dimensional meshes.

    Arguments:

        - m is a dolfin function on a 1d or 2d vector function space
        - dim is the dimension of the dmi
   
    On 3-dimensional meshes, dolfin supports computing the integrand
    of the DMI energy using

         df.inner(m, df.curl(m))        eq.1

    However, the curl operator is not implemented on 1d and 2d meshes.
    With the expansion of the curl in cartesian coordinates

        curlx = dmzdy - dmydz
        curly = dmxdz - dmzdx
        curlz = dmydx - dmxdy

    we can compute eq. 1 with

        (mx * curlx + my * curly + mz * curlz).

    """
    if not dim in (1, 2):
        raise ValueError("dim must be 1 or 2, "
            "don't use this on higher-dimensional meshes")

    gradm = df.grad(m)

    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]

    if dim == 1:
        # there are no derivatives in y-direction on a 1d
        # mesh so we can set them to zero
        dmxdy = 0
        dmydy = 0
        dmzdy = 0
    elif dim == 2:
        dmxdy = gradm[0, 1]
        dmydy = gradm[1, 1]
        dmzdy = gradm[2, 1]

    # there are no derivatives along z on either 1d or 2d
    # meshes, so we can set them to zero
    dmxdz = 0
    dmydz = 0
    dmzdz = 0

    curlx = dmzdy - dmydz
    curly = dmxdz - dmzdx
    curlz = dmydx - dmxdy
    return (m[0] * curlx + m[1] * curly + m[2] * curlz)



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
            
            
        dmi_type
            options are 'auto', '1d', '2d', '3d' and 'interfacial', default is 'auto' 
            which means the dmi is auto selected according to the mesh dimension.


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

    def __init__(self, D, method="box-matrix-petsc", name='DMI', dmi_type='auto'):
        self.D_waiting_for_mesh = D
        self.name = name
        super(DMI, self).__init__(method, in_jacobian=True)
        self.dmi_type = dmi_type

    @mtimed
    def setup(self, S3, m, Ms, unit_length=1):
        self.DMI_factor = df.Constant(1.0 / unit_length)
        self.S3 = S3
        self.D = helpers.scalar_valued_dg_function(self.D_waiting_for_mesh,
                                                   self.S3.mesh())
        self.D.rename('D', 'DMI_constant')
        self.D_av = np.average(self.D.vector().array())
        del(self.D_waiting_for_mesh)

        meshdim = S3.mesh().topology().dim()
        
        dmi_dim  = meshdim
        
        if self.dmi_type is '1d':
            dmi_dim = 1
        elif self.dmi_type is '2d':
            dmi_dim = 2
        elif self.dmi_type is '3d':
            dmi_dim = 3
        
        if dmi_dim == 3:
            E_integrand = self.DMI_factor * self.D * df.inner(m, df.curl(m))
        else:
            E_integrand = self.DMI_factor * self.D * manual_times_curl(m, dmi_dim)
            
        if self.dmi_type is 'interfacial':
            E_integrand = DMI_ultra_thin_film(m, self.DMI_factor * self.D, dim=dmi_dim)

        super(DMI, self).setup(E_integrand, S3, m, Ms, unit_length)


def DMI_ultra_thin_film(m, D, dim):
    """Input arguments:

       m a dolfin 3d-vector function on a 1d or 2d space,
       D is the DMI constant
       dim is the mesh dimension.

       Returns the form to compute the DMI energy:

         D (m_x * dm_z/dx - m_z * dm_x/dx) +
         D (m_y * dm_z/dy - m_z * dm_y/dy) * df.dx                      (1)

    Equation is equation (2) from the Oct 2013 paper by Rohart
    and Thiaville (http://arxiv.org/abs/1310.0666)

    """
    #in principle, this class also works for 3d mesh? 
    #assert dim <= 2, "Only implemented for 2d mesh"

    gradm = df.grad(m)

    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]

    if dim == 1:
        dmxdy = 0
        dmydy = 0
        dmzdy = 0
    else: # works for 2d mesh or 3d mesh
        dmxdy = gradm[0, 1]
        dmydy = gradm[1, 1]
        dmzdy = gradm[2, 1]


    mx = m[0]
    my = m[1]
    mz = m[2]

    return D * (mx * dmzdx - mz * dmxdx) + D * (my * dmzdy - mz * dmydy)
