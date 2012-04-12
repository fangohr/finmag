import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.helpers import components

length = 20e-9 # m
simplices = 10
mesh = df.Interval(simplices, 0, length)

bigmesh = df.Interval(1000, 0, length)

def test_there_should_be_no_exchange_field_for_uniform_M():
    llg = LLG(mesh)
    llg.set_m((llg.Ms, 0, 0))
    llg.setup()
    llg.solve()
    H_ex = llg.H_ex
    print "max(H_ex)=%g" % (max(abs(H_ex)))
    if llg.exchange.method=='box-assemble':
        assert np.array_equal(H_ex, np.zeros(len(H_ex)))
    elif llg.exchange.method in ['box-matrix-numpy','box-matrix-petsc'] :
        assert max(abs(H_ex)) < 5e-9 #The pre-assembled matrix method is faster but less accurate.
    else:
        assert np.array_equal(H_ex, np.zeros(len(H_ex))) #this may fail -- we never tested any other method

def test_there_should_be_an_exchange_field_for_heterogeneous_M():
    llg = LLG(mesh)
    llg.set_m((
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0'), L=length)
    llg.setup()
    llg.solve()
    H_ex = llg.H_ex
    assert not np.array_equal(H_ex, np.zeros(len(H_ex)))

def test_exchange_field_should_change_when_M_changes():
    llg = LLG(mesh)
    llg.set_m((
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0'), L=length)
    llg.setup()

    # save the beginning value of M for comparison purposes
    old_m = llg.m

    # solve the LLG, and update the magnetisation.
    dt = 1e-11
    dMdt = llg.solve()
    llg.m = llg.m + dMdt*dt

    # Capture the current value of the exchange field.
    old_H_ex = llg.H_ex[:]
    # We assert that the magnetisation has indeed changed since the beginning.
    assert not np.array_equal(old_m, llg.m)
    # If we now solve the LLG again, we expect the new value of the
    # exchange field to change (because the magnetisation has changed).
    new_H_ex = llg.exchange.compute_field()
    assert not np.array_equal(old_H_ex, new_H_ex), "H_ex hasn't changed."

def test_exchange_field_box_assemble_equal_box_matrix():
    """Simulation 1 is computing H_ex=dE_dM via assemble.
    Simulation 2 is computing H_ex=g*M with a suitable pre-computed matrix g.
    
    Here we show that the two methods give equivalent results.
    """
    m_initial = (
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0')
    llg1 = LLG(mesh)
    llg1.set_m(m_initial, L=length)
    llg1.setup(exchange_method='box-matrix-numpy')
    llg1.solve()
    H_ex1 = llg1.H_ex

    llg2 = LLG(mesh)
    llg2.set_m(m_initial, L=length)
    llg2.setup(exchange_method='box-assemble')
    llg2.solve()
    H_ex2 = llg2.H_ex

    diff = max(abs(H_ex1-H_ex2))
    print "Difference between H_ex1 and H_ex2: max(abs(H_ex1-H_ex2))=%g" % diff
    print "Max value = %g, relative error = %g " % (max(H_ex1), diff/max(H_ex1))
    assert diff < 1e-8
    assert diff/max(H_ex1)<1e-15


def test_exchange_field_box_matrix_numpy_same_as_box_matrix_petsc():
    """Simulation 1 is computing H_ex=dE_dM via assemble.
    Simulation 2 is computing H_ex=g*M with a suitable pre-computed matrix g.
    
    Here we show that the two methods give equivalent results.
    """
    m_initial = (
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0')
    llg1 = LLG(mesh)
    llg1.set_m(m_initial, L=length)
    llg1.setup(exchange_method='box-matrix-numpy')
    llg1.solve()
    H_ex1 = llg1.H_ex

    llg2 = LLG(mesh)
    llg2.set_m(m_initial, L=length)
    llg2.setup(exchange_method='box-matrix-petsc')
    llg2.solve()
    H_ex2 = llg2.H_ex

    diff = max(abs(H_ex1-H_ex2))
    print "Difference between H_ex1 and H_ex2: max(abs(H_ex1-H_ex2))=%g" % diff
    print "Max value = %g, relative error = %g " % (max(H_ex1), diff/max(H_ex1))
    assert diff < 1e-8
    assert diff/max(H_ex1)<1e-15

if __name__=="__main__":
    test_exchange_field_box_matrix_numpy_same_as_box_matrix_petsc()
    test_exchange_field_box_assemble_equal_box_matrix()
    test_exchange_field_should_change_when_M_changes()
    test_exchange_field_box_assemble_equal_box_matrix()
