import dolfin as df
import numpy as np
from finmag.util.consts import mu0
from finmag import sim_with, Field

np.set_printoptions(precision=2)


def generate_random_vectors(num=10, length=1.0):
    v = np.random.random_sample((num, 3))
    v_norms = np.linalg.norm(v, axis=1)
    return length * v / v_norms[:, np.newaxis]


class TestZeeman(object):
    @classmethod
    def setup_class(cls):
        """
        Create a box mesh and a simulation object on this mesh which
        will be used to compute Zeeman energies in the individual tests.

        """
        # The mesh and simulation are only created once for the entire
        # test class and are re-used in each test method for efficiency.
        cls.Lx, cls.Ly, cls.Lz = 100, 30, 10
        nx, ny, nz = 30, 10, 5
        mesh = df.BoxMesh(0, 0, 0, cls.Lx, cls.Ly, cls.Lz, nx, ny, nz)
        unit_length = 1e-9
        cls.mesh_vol = cls.Lx * cls.Ly * cls.Lz  * unit_length**3
        cls.S3 = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)

        # The values for Ms and m_init are arbitrary as they will be
        # ovewritten in the tests.
        cls.sim = sim_with(mesh, Ms=1, m_init=(1, 0, 0), demag_solver=None,
                           unit_length=unit_length)

    @classmethod
    def set_simulation_parameters(cls, M, H):
        """
        Sett m, Ms and H on the pre-created simulation object. This should
        be done before calling any of the other functions which compute the
        energy, energy density, etc. using Finmag.

        """
        Ms = np.linalg.norm(M)
        m = M / Ms

        cls.sim.m = m
        cls.sim.Ms = Ms
        cls.sim.set_H_ext(H)

    @classmethod
    def compute_energy_with_finmag(cls):
        """
        Compute Zeeman energy of the pre-created simulation using Finmag.

        """
        return cls.sim.get_interaction('Zeeman').compute_energy()

    @classmethod
    def compute_energy_density_with_finmag(cls):
        """
        Compute Zeeman energy density of the pre-created simulation
        using Finmag.

        """
        return cls.sim.get_interaction('Zeeman').energy_density()

    @classmethod
    def compute_field_with_finmag(cls):
        """
        Compute Zeeman field of the pre-created simulation using Finmag.

        """
        return cls.sim.get_interaction('Zeeman').H

    @classmethod
    def create_linearly_varying_field(cls, H0, H1):
        """
        Return a Field which varies linearly from H0 at mesh corner (0, 0, 0)
        to H1 at mesh corner (cls.Lx, cls.Ly, cls.Lz).
        """
        H_field = Field(cls.S3,
            df.Expression(
                ['(1-x[0]/Lx)*H0x + x[0]/Lx*H1x',
                 '(1-x[1]/Ly)*H0y + x[1]/Ly*H1y',
                 '(1-x[2]/Lz)*H0z + x[2]/Lz*H1z'],
                H0x=H0[0], H0y=H0[1], H0z=H0[2],
                H1x=H1[0], H1y=H1[1], H1z=H1[2],
                Lx=cls.Lx, Ly=cls.Ly, Lz=cls.Lz))

        return H_field

    def test_total_energy_with_constant_field(self):
        """
        Check Zeeman energy for some random (but spatially uniform)
        values of magnetisation M and external field H. The energy
        computed with Finmag is compared to the value obtained from
        an analytical expression.

        """
        Ms = 8e5  # saturation magnetisation
        H_norm = 3e4  # strength of external field

        num = 3
        M_vals = generate_random_vectors(num, length=Ms)
        H_vals = generate_random_vectors(num, length=H_norm)

        for M in M_vals:
            for H in H_vals:
                print("Comparing energy for M={}, H={}".format(M, H))
                self.set_simulation_parameters(M, H)

                energy_expected = -mu0 * self.mesh_vol * np.dot(M, H)
                energy_finmag = self.compute_energy_with_finmag()

                energy_density_expected = -mu0 * np.dot(M, H)
                energy_density_finmag = self.compute_energy_density_with_finmag().get_ordered_numpy_array()

                zeeman_field_expected = Field(self.S3, H)
                zeeman_field_computed = self.compute_field_with_finmag()

                np.testing.assert_allclose(energy_expected, energy_finmag)
                np.testing.assert_allclose(energy_density_expected, energy_density_finmag)
                assert zeeman_field_expected.allclose(zeeman_field_computed)

    def test_total_energy_linearly_varying_field(self):
        """
        Check Zeeman energy for some random (but spatially uniform) values
        of magnetisation M and external field H, where M is spatially
        uniform but H varies linearly between opposite corners of the
        cuboid mesh. The energy computed with Finmag is compared to
        the value obtained from an analytical expression.

        """
        Ms = 8e5  # saturation magnetisation
        H_norm = 3e4  # strength of external field

        num = 3
        M_vals = generate_random_vectors(num, length=Ms)
        H0_vals = generate_random_vectors(num, length=H_norm)
        H1_vals = generate_random_vectors(num, length=H_norm)

        for M in M_vals:
            for H0, H1 in zip(H0_vals, H1_vals):
                print("Comparing energy for M={}, H0={}, H1={}".format(M, H0, H1))
                H_field = self.create_linearly_varying_field(H0, H1)
                M_field = Field(self.S3, M)
                self.set_simulation_parameters(M, H_field)

                energy_expected = -mu0 * self.mesh_vol * np.dot((H0 + H1)/2, M)
                energy_finmag = self.compute_energy_with_finmag()
                np.testing.assert_allclose(energy_expected, energy_finmag)

                energy_density_expected = -mu0 * M_field.dot(H_field)
                energy_density_finmag = self.compute_energy_density_with_finmag()

                zeeman_field_expected = H_field
                zeeman_field_computed = self.compute_field_with_finmag()

                np.testing.assert_allclose(energy_expected, energy_finmag)
                assert energy_density_expected.allclose(energy_density_finmag)
                assert zeeman_field_expected.allclose(zeeman_field_computed)
