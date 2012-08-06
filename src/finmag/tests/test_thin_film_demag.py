import numpy as np
import dolfin as df
from finmag import Simulation as Sim
from finmag.energies import ThinFilmDemag

TOLERANCE = 1e-15
Ms = 8.6e5

def test_zero_thin_film_demag():
    mesh = df.UnitCube(2, 2, 2)
    sim = Sim(mesh, Ms)
    sim.set_m((1, 0, 0))

    demag = ThinFilmDemag()
    sim.add(demag)
    H = demag.compute_field()

    expected_H = np.zeros(sim.m.shape)
    diff = np.abs(H - expected_H)
    print "Expected, with shape {}:\n".format(expected_H.shape), expected_H
    print "Got, with shape {}:\n".format(H.shape), H
    print "Difference:\n", diff
    assert np.max(diff) < TOLERANCE

def test_thin_film_demag():
    mesh = df.UnitCube(2, 2, 2)
    sim = Sim(mesh, Ms)
    sim.set_m((0, 0, 1))

    demag = ThinFilmDemag()
    sim.add(demag)
    H = demag.compute_field()

    expected_H = - Ms * sim.m
    diff = np.abs(H - expected_H)
    print "Expected, with shape {}:\n".format(expected_H.shape), expected_H
    print "Got, with shape {}:\n".format(H.shape), H
    print "Difference:\n", diff
    assert np.max(diff) < TOLERANCE
