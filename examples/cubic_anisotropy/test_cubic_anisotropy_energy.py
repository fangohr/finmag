import dolfin as df
from finmag.energies import CubicAnisotropy

Ms = 876626  # A/m

K1 = -8608726
K2 = -13744132
K3 =  1100269
u1 = (0, -0.7071, 0.7071)
u2 = (0,  0.7071, 0.7071)
u3 = (-1, 0, 0)  # perpendicular to u1 and u2


def test_cubic_anisotropy_energy():
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 40, 1, 1, 40)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)

    m = df.Function(S3)
    m.assign(df.Constant((0, 0, 1)))

    ca = CubicAnisotropy(u1, u2, K1, K2, K3)
    ca.setup(S3, m, Ms, unit_length=1e-9)

    energy = ca.compute_energy()
    energy_expected = 8.3e-20  # oommf cubicEight_100pc.mif -> ErFe2.odt

    rel_diff = abs(energy - energy_expected) / abs(energy_expected)
    assert rel_diff < 1e-2
