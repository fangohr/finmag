import numpy as np
import dolfin as df
from scipy.integrate import odeint
from finmag.sim.llg import LLG
from finmag.sim.helpers import components

length = 20e-9 # m
simplices = 10
mesh = df.Interval(simplices, 0, length)

def test_when_interpolating_M_you_need_to_define_the_problem_again():
    """ this test documents the behaviour of the LLG class and
    may be removed once it improves. """

    llg = LLG(mesh)
    llg.set_m0(('(2*x[0]-L)/L',
                'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
                '0'), L=length)
    llg.setup()
    llg.solve()
    H_ex = llg.H_ex[:]

    llg.set_m0(('sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
                '(2*x[0]-L)/L',
                '0'), L=1e-5)
    # not doing the setup again.
    llg.solve()
    new_H_ex = llg.H_ex

    # even though M has a new value the computation of M refers to the old one
    assert np.array_equal(H_ex, new_H_ex)

def test_updating_the_M_vector_is_okay_though():
    llg = LLG(mesh)
    llg.set_m0((
        '(2*x[0]/L - 1)',
        'sqrt(1 - (2*x[0]/L - 1)*(2*x[0]/L - 1))',
        '0'), L=length)
    llg.setup()
    llg.solve()
    old_m = llg.m
    old_H_ex = llg.H_ex[:]

    # provide a new magnetisation, without calling df.interpolate
    llg.m = np.zeros(len(llg.m))
    new_m = llg.m
    llg.solve()
    new_H_ex = llg.H_ex[:]
    assert not np.array_equal(old_m, new_m)
    assert not np.array_equal(old_H_ex, new_H_ex)

def test_there_should_be_no_exchange_field_for_uniform_M():
    llg = LLG(mesh)
    llg.set_m0((llg.Ms, 0, 0))
    llg.setup()
    llg.solve()
    H_ex = llg.H_ex
    assert np.array_equal(H_ex, np.zeros(len(H_ex)))

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

def test_method_of_computing_the_average_matters():
    llg = LLG(mesh)
    llg.set_m0((
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0'), L=length)
    llg.setup()
    
    averages = np.mean(components(llg.m), axis=1)
    diff = np.abs(llg.m_average - averages)
    assert diff.max() > 5e-2
