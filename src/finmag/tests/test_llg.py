import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.helpers import components

length = 20e-9 # m
simplices = 10
mesh = df.Interval(simplices, 0, length)

def test_updating_the_M_vector_is_okay_though():
    llg = LLG(mesh)
    llg.set_m((
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


def test_method_of_computing_the_average_matters():
    llg = LLG(mesh)
    llg.set_m((
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0'), L=length)
    llg.setup()
    
    averages = np.mean(components(llg.m), axis=1)
    diff = np.abs(llg.m_average - averages)
    assert diff.max() > 5e-2

def test_spatially_varying_alpha():
    llg = LLG(mesh)

    # one value of alpha on all nodes
    llg.alpha = 0.5 # API hasn't changed.
    expected_alphas = 0.5 * np.ones(simplices+1)
    assert np.array_equal(llg.alpha_vec, expected_alphas) 

    # spatially varying alpha
    multiplicator = df.Function(llg.F)
    multiplicator.vector()[:] = np.linspace(0.5, 1.5, simplices+1)
    llg.spatially_varying_alpha(0.4, multiplicator) # This is new.
    expected_alphas = np.linspace(0.2, 0.6, simplices+1)
    print "Got:\n", llg.alpha_vec
    print "Expected:\n", expected_alphas
    assert np.allclose(llg.alpha_vec, expected_alphas) 
