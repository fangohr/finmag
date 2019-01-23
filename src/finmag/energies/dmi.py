import logging
import dolfin as df
from aeon import timer
from finmag.field import Field
from energy_base import EnergyBase
from finmag.util.helpers import times_curl

logger = logging.getLogger('finmag')

class DMI(EnergyBase):
    """
    Compute the Dzyaloshinskii-Moriya Interaction (DMI) field.
    .. math::
        E_{\\text{DMI}} = \\int_\\Omega D \\vec{m} \\cdot
                          (\\nabla \\times \\vec{m}) dx
    *Arguments*
        D
            the DMI constant
        method
            See documentation of EnergyBase class for details.
        dmi_type
            Options are 'auto', '1d', '2d', '3d' and 'interfacial'.
            Default value is 'auto' which means the dmi is automaticaly
            selected according to the mesh dimension.
    *Example of Usage*
        .. code-block:: python
            import dolfin as df
            from finmag.energies.dmi import DMI
            from finmag.field import Field
            # Define a mesh representing a cube with edge length L.
            L = 1e-8  # m
            n = 5
            mesh = df.BoxMesh(df.Point(0, L, 0), df.Point(L, 0, L), n, n, n)
            D = 5e-3  # J/m**2 DMI constant
            Ms = 0.8e6  # A/m magnetisation saturation
            # Initial magnetisation
            S3 = df.VectorFunctionSpace(mesh, 'CG', 1)
            m = Field(S3, (1, 0, 0))
            dmi = DMI(D)
            dmi.setup(m, Ms)
            # Compute DMI energy.
            E_dmi = dmi.compute_energy()
            # Compute DMI effective field.
            H_dmi = dmi.compute_field()
            # Using 'box-matrix-numpy' method (fastest for small matrices)
            dmi_np = DMI(D, method='box-matrix-numpy')
            dmi_np.setup(m, Ms)
            H_dmi_np = dmi_np.compute_field()
    """

    def __init__(self, D, method='box-matrix-petsc', name='DMI',
                 dmi_type='auto'):
        self.D_value = D  # Value of D, later converted to a Field object.
        self.name = name
        self.dmi_type = dmi_type

        super(DMI, self).__init__(method, in_jacobian=True)


    @timer.method
    def setup(self, m, Ms, unit_length=1):
        # Create an exchange constant Field object A in DG0 function space.
        dg_functionspace = df.FunctionSpace(m.mesh(), 'DG', 0)
        self.D = Field(dg_functionspace, self.D_value, name='D')
        del(self.D_value)

        # Multiplication factor used for dmi energy computation.
        self.dmi_factor = df.Constant(1.0/unit_length)

        if self.dmi_type is '1d':
            dmi_dim = 1
        elif self.dmi_type is '2d':
            dmi_dim = 2
        elif self.dmi_type is '3d':
            dmi_dim = 3
        else:
            dmi_dim = m.mesh_dim()

        # Select the right expression for computing the dmi energy.
        if self.dmi_type is 'interfacial':
            E_integrand = DMI_interfacial(m, self.dmi_factor*self.D.f,
                                          dim=dmi_dim)
        elif self.dmi_type is 'D2D':
            E_integrand = DMI_D2D(m, self.dmi_factor*self.D.f,
                                  dim=dmi_dim)
        else:
            E_integrand = self.dmi_factor*self.D.f*times_curl(m.f, dmi_dim)

        super(DMI, self).setup(E_integrand, m, Ms, unit_length)

        if self.method == 'direct':
            self.__setup_field_direct()

    def __setup_field_direct(self):
        dofmap = self.m.mesh_dofmap()
        S3 = df.VectorFunctionSpace(self.m.mesh(), "CG", 1, dim=3,
                                   constrained_domain=dofmap.constrained_domain)

        u3 = df.TrialFunction(S3)
        v3 = df.TestFunction(S3)
        self.g_petsc = df.PETScMatrix()
        df.assemble(-2*self.dmi_factor*self.D.f*df.inner(v3, df.curl(u3))*df.dx, tensor=self.g_petsc)
        self.H_petsc = df.PETScVector()

def DMI_interfacial(m, D, dim):
    """
    Input arguments:
    m is a Field object on a 1d or 2d space,
    D is the DMI constant
    dim is the mesh dimension.
    Returns the form to compute the DMI energy:
    D(m_x * dm_z/dx - m_z * dm_x/dx) +
    D(m_y * dm_z/dy - m_z * dm_y/dy) * df.dx
    References:
    [1] Rohart, S. and Thiaville A., Phys. Rev. B 88, 184422 (2013)
    """

    gradm = df.grad(m.f)

    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]

    if dim == 1:
        dmxdy = 0
        dmydy = 0
        dmzdy = 0
    else:
        # Works for both 2d mesh or 3d meshes.
        dmxdy = gradm[0, 1]
        dmydy = gradm[1, 1]
        dmzdy = gradm[2, 1]

    mx = m.f[0]
    my = m.f[1]
    mz = m.f[2]

    return D*(mx * dmzdx - mz * dmxdx) + D*(my * dmzdy - mz * dmydy)

def DMI_D2D(m, D, dim):
    """
    D2D type DMI

    w = D (L_xz^y + L_yz^x)

    where L_ij^k = m_i dm_j / dk - m_j dm_i / dk

    Hence:

    w = D [(m_x dm_z / dy - m_z dm_x / dy) -
           (m_y dm_z / dx - m_z dm_y / dx)]

      = D (m_x dm_z/dy + m_y dm_z/dx - m_z ( dm_x / dy + dm_y / dx))

    """

    if dim != 3:
        raise ValueError("This DMI Type does not work other than in 3-D.")

    gradm = df.grad(m.f)

    dmxdx = gradm[0, 0]
    dmxdy = gradm[0, 1]
    dmzdx = gradm[2, 0]
    dmzdy = gradm[2, 1]

    mx = m.f[0]
    my = m.f[1]
    mz = m.f[2]

    return D * (mx * dmzdy + my * dmzdx - mz * ( dmxdy + dmydx))
