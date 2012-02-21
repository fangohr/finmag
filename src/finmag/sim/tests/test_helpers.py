import numpy as np
from finmag.sim.helpers import *

TOLERANCE = 1e-15

def test_components():
    x = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
    y = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert np.array_equal(y, components(x))

def test_vectors():
    x1 = np.array([1, 1, 2, 2, 3, 3])
    y1 = np.array([[1, 2, 3], [1, 2, 3]])
    assert np.array_equal(y1, vectors(x1))

    x2 = np.array([0, 1, 1, 0, 2, 3, 3, 2, 4, 5, 5, 4])
    y2 = np.array([[0, 2, 4], [1, 3, 5], [1, 3, 5], [0, 2, 4]])
    assert np.array_equal(y2, vectors(x2))

def test_norm():
    assert norm([0, 0, 0]) == 0
    assert norm([1, 0, 0]) == 1
    assert norm([1, 1, 0]) - np.sqrt(2) < TOLERANCE
    assert norm([1, 1, 1]) - np.sqrt(3) < TOLERANCE
    assert norm([-1, 0, 0]) - norm([1, 0, 0]) < TOLERANCE 
    assert 3*norm([1, 1, 1]) - norm(3*numpy.array([1, 1, 1])) < TOLERANCE

def test_angle():
    assert angle([1,0,0],[1,0,0])           < TOLERANCE
    assert angle([1,0,0],[0,1,0]) - np.pi/2 < TOLERANCE
    assert angle([1,0,0],[1,1,0]) - np.pi/4 < TOLERANCE

def test_rows_to_columns():
    x = numpy.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    y = numpy.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    assert numpy.array_equal(y, rows_to_columns(x))

def test_perturbed_vectors():
    quantity = 10; direction = [1, 0, 0]; length = 5
    # I could pass a fake random function to perturbed_vectors for testing,
    # but the exact behaviour of perturbed_vectors is unspecified except
    # for the number of vectors returned and their length anyways.

    vector_field = perturbed_vectors(quantity, direction, length)
    assert len(vector_field) == 10

    for v in vector_field:
        assert len(v) == 3
        assert norm(v) - 5 < TOLERANCE
