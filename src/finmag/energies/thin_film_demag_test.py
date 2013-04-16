import numpy as np
import dolfin as df
from finmag import Simulation as Sim
from finmag.energies import ThinFilmDemag, Demag

Ms = 8.6e5


def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    demag = ThinFilmDemag()
    assert hasattr(demag, 'name')


def compare_with_demag_from_initial_m(H_gen, m_init, atol=0, rtol=0):
    sim = Sim(df.UnitCubeMesh(2, 2, 2), Ms)
    sim.set_m(m_init)

    demag = ThinFilmDemag()
    sim.add(demag)
    H_computed = demag.compute_field()
    H_expected = H_gen(sim.m)

    diff = np.abs(H_computed - H_expected)
    print "Expected, with shape {}:\n".format(H_expected.shape), H_expected
    print "Got, with shape {}:\n".format(H_computed.shape), H_computed
    print "Difference:\n", diff
    assert np.allclose(H_computed, H_expected, atol=atol, rtol=rtol)


def test_zero_thin_film_demag():
    compare_with_demag_from_initial_m(lambda m: np.zeros(m.shape), (1, 0, 0), atol=1e-14)
    compare_with_demag_from_initial_m(lambda m: np.zeros(m.shape), (1, 1, 0), atol=1e-14)


def test_thin_film_demag():
    compare_with_demag_from_initial_m(lambda m: -Ms * m, (0, 0, 1), rtol=1e-14)


def test_thin_film_demag_against_real_demag():
    sim = Sim(df.BoxMesh(0, 0, 0, 500e-9, 500e-9, 1e-9, 50, 50, 1), Ms)
    sim.set_m((0, 0, 1))

    tfdemag = ThinFilmDemag()
    sim.add(tfdemag)
    H_tfdemag = tfdemag.compute_field().view().reshape((3, -1)).mean(1)
    demag = Demag()
    sim.add(demag)
    H_demag = demag.compute_field().view().reshape((3, -1)).mean(1)

    diff = np.abs(H_tfdemag - H_demag)/Ms
    print "Standard Demag:\n", H_demag
    print "ThinFilmDemag:\n", H_tfdemag
    print "Difference relative to Ms:\n", diff
    assert np.allclose(H_tfdemag, H_demag, atol=0.05*Ms)  # 5% of Ms

    sim.set_m((1, 0, 0))
    H_tfdemag = tfdemag.compute_field().view().reshape((3, -1)).mean(1)
    H_demag = demag.compute_field().view().reshape((3, -1)).mean(1)

    print "Running again, changed m in the meantime."
    diff = np.abs(H_tfdemag - H_demag)/Ms
    print "Standard Demag:\n", H_demag
    print "ThinFilmDemag:\n", H_tfdemag
    print "Difference relative to Ms:\n", diff
    assert np.allclose(H_tfdemag, H_demag, atol=0.005*Ms)  # 0.5% of Ms
