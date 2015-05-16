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
        elif self.dmi_type in ['C_nv', 'D_n', 'D_2d', 'C_n', 'S_4']:
            E_integrand = LI_assembler(m, self.dmi_factor*self.D.f, self.dmi_type)
        else:
            E_integrand = self.dmi_factor*self.D.f*times_curl(m.f, dmi_dim)

        super(DMI, self).setup(E_integrand, m, Ms, unit_length)

def LI_assembler(m, D, crystalsym):
    """
    This is the LI assembler for the terms as listed in ANB89.

    Remember: "To stabilize vortex states, an additional
    uniaxial anisotropy must be present." (Bogdanov 1994)

    Also: Note the comments below on the exclusion of certain lifshitz
    invariants.
    """
    #logger.info("Running LI assembler with type ", crystalsym)
    gradm = df.grad(m.f)

    # TODO How costly are these assignments?
    dmxdx = gradm[0, 0]
    dmydx = gradm[1, 0]
    dmzdx = gradm[2, 0]
    dmxdy = gradm[0, 1]
    dmydy = gradm[1, 1]
    dmzdy = gradm[2, 1]
    # Not in use currently (see below):
    #dmxdz = gradm[0, 2]
    #dmydz = gradm[1, 2]
    #dmzdz = gradm[2, 2]

    mx = m.f[0]
    my = m.f[1]
    mz = m.f[2]

    # TODO (Bogdanov 1989) or (1994) notation?
    w_1 = mz*dmxdx - mx*dmzdx + mz*dmydy - my*dmzdy
    w_2 = mz*dmxdy - mx*dmzdy - mz*dmydx + my*dmzdx
    # (Note: ANB89 uses w_1' and w_2' instead.)
    w_3 = mz*dmxdx - mx*dmzdx - mz*dmydy + my*dmzdy
    w_4 = mz*dmxdy - mx*dmzdy + mz*dmydx - my*dmzdx
    #w_z = mx*dmydz - my*dmxdz  # w_3 in (Bogdanov 1994).

    # Note: All terms w_z are left out!  ANB89: "We note also that the
    # classes D_n and C_n admit the invariants [w_z] which can lead to
    # the formation of a spiral structure with propagation vector along
    # the z axis." (Bogdanov, Hubert 1994) says the same thing.

    # Nota bene, the DMI_interfacial() below implements -w_1 (C_nv).

    # TODO w_1...4 use actually only 4 distinct LIs:
    # LIs, spelled out: m_i delta_j m_k - m_k delta_j m_i
    #
    #w_1 = w_11 + w_12    w_2 = w_21 + w_22
    #w_3 = w_11 - w_12    w_4 = w_21 - w_22

    # For now, I'll equate all different dmi constants.
    a = a_1 = a_2 = a_3 = a_4 = a_5 = a_6 = D
    # Note: At least C_nv and D_n admit w_z terms (see above)!
    C_nv = a * w_1
    D_n  = a_1 * w_2
    D_2d = a_2 * w_4
    C_n  = a_3 * w_1 + a_4 * w_2
    S_4  = a_5 * w_3 + a_6 * w_4

    # TODO I'd really like a more elegant dispatch table here.  I want
    # many other pretty things, too.
    if crystalsym is 'C_nv':   return C_nv
    elif crystalsym is 'D_n':  return D_n
    elif crystalsym is 'D_2d': return D_2d
    elif crystalsym is 'C_n':  return C_n
    elif crystalsym is 'S_4':  return S_4
    else: True # TODO Die?  Do nothing?  What?

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
