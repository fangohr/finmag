import dolfin as df
from finmag import Simulation
from finmag.energies import CubicAnisotropy

Ms = 876626  # A/m

K1 = -8608726
K2 = -13744132
K3 =  1100269
u1 = (0, -0.7071, 0.7071)
u2 = (0,  0.7071, 0.7071)


def test_cubic_anisotropy_energy():
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 40, 1, 1, 40)
    sim = Simulation(mesh, Ms, unit_length=1e-9)
    sim.set_m((0, 0, 1))

    ca = CubicAnisotropy(u1, u2, K1, K2, K3)
    sim.add(ca)

    energy = ca.compute_energy()
    print "Computed cubic anisotropy energy. E = {}.".format(energy)
    energy_expected = 8.3e-20  # oommf cubicEight_100pc.mif -> ErFe2.odt
    print "Expected E = {}.".format(energy_expected)

    rel_diff = abs(energy - energy_expected) / abs(energy_expected)
    assert rel_diff < 1e-2
