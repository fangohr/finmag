import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.helpers import components

length = 20e-9 # m
simplices = 10
mesh = df.Interval(simplices, 0, length)

def test_there_should_be_no_exchange_field_for_uniform_M():
    llg = LLG(mesh)
    llg.set_m0((llg.Ms, 0, 0))
    llg.setup()
    llg.solve()
    H_ex = llg.H_ex
    print "max(H_ex)=%g" % (max(abs(H_ex)))
    if llg.exchange.method=='box-assemble':
        assert np.array_equal(H_ex, np.zeros(len(H_ex)))
    elif llg.exchange.method=='box':
        assert max(abs(H_ex)) < 5e-9 #The pre-assembled matrix method is faster but less accurate.
    else:
        assert np.array_equal(H_ex, np.zeros(len(H_ex))) #this may fail -- we never tested any other method

def test_there_should_be_an_exchange_field_for_heterogeneous_M():
    llg = LLG(mesh)
    llg.set_m0((
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0'), L=length)
    llg.setup()
    llg.solve()
    H_ex = llg.H_ex
    assert not np.array_equal(H_ex, np.zeros(len(H_ex)))

def test_exchange_field_should_change_when_M_changes():
    llg = LLG(mesh)
    llg.set_m0((
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

