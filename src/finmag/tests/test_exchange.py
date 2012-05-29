import numpy as np
import dolfin as df
from finmag.energies import Exchange
from finmag import Simulation as Sim

length = 20e-9 # m
simplices = 10
mesh = df.Interval(simplices, 0, length)
S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
Ms = 8.6e5
A = 1.3e-11

bigmesh = df.Interval(1000, 0, length)

def test_there_should_be_no_exchange_field_for_uniform_M():
    m = df.interpolate(df.Constant((1, 0, 0)), S3)

    exchange = Exchange(A)
    exchange.setup(S3, m, Ms)
    H_ex = exchange.compute_field()

    print "max(H_ex)=%g" % (max(abs(H_ex)))
    if exchange.method=='box-assemble':
        assert np.array_equal(H_ex, np.zeros(len(H_ex)))
    elif exchange.method in ['box-matrix-numpy','box-matrix-petsc'] :
        assert max(abs(H_ex)) < 5e-9 #The pre-assembled matrix method is faster but less accurate.
    else:
        assert np.array_equal(H_ex, np.zeros(len(H_ex))) #this may fail -- we never tested any other method

def test_there_should_be_an_exchange_field_for_heterogeneous_M():
    m_expr = df.Expression(
        ('(2*x[0]-L)/L',
         'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
         '0'), L=length)
    m = df.interpolate(m_expr, S3)

    exchange = Exchange(A)
    exchange.setup(S3, m, Ms)
    H_ex = exchange.compute_field()
    assert not np.array_equal(H_ex, np.zeros(len(H_ex)))

def test_exchange_field_should_change_when_M_changes():
    sim = Sim(mesh, Ms)
    sim.set_m(df.Expression(
        ('(2*x[0]-L)/L',
         'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
         '0'), L=length))

    exchange = Exchange(A)
    sim.add(exchange)

    # save the beginning value of M and the exchange field for comparison purposes
    old_m = sim.m
    old_H_ex = exchange.compute_field()

    sim.run_until(1e-11)

    # Capture the current value of the exchange field and m.
    m = sim.m
    H_ex = exchange.compute_field() 

    # We assert that the magnetisation has indeed changed since the beginning.
    assert not np.array_equal(old_m, m)
    assert not np.array_equal(old_H_ex, H_ex), "H_ex hasn't changed."

def test_exchange_field_equivalent_methods():
    """
    Simulation 1 is computing H_ex=dE_dM via assemble.
    Simulation 2 is computing H_ex=g*M with a suitable pre-computed matrix g.
    Simulation 3 computes g as a petsc matrix.
    
    Here we show that the three methods give equivalent results.

    """
    m_expr = df.Expression(
        ('(2*x[0]-L)/L',
         'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
         '0'), L=length)
    m = df.interpolate(m_expr, S3)

    exchange = Exchange(A, method="box-assemble")
    exchange.setup(S3, m, Ms)
    H_ex_1 = exchange.compute_field()
    exchange = Exchange(A, method="box-matrix-numpy")
    exchange.setup(S3, m, Ms)
    H_ex_2 = exchange.compute_field()
    exchange = Exchange(A, method="box-matrix-petsc")
    exchange.setup(S3, m, Ms)
    H_ex_3 = exchange.compute_field()

    diff12 = max(abs(H_ex_1-H_ex_2))
    diff13 = max(abs(H_ex_1-H_ex_3))
    print "Difference between H_ex1 and H_ex2: max(abs(H_ex1-H_ex2))=%g" % diff12
    print "Max value = %g, relative error = %g " % (max(H_ex_1), diff12/max(H_ex_1))
    assert diff12 < 1e-8
    assert diff12/max(H_ex_1)<1e-15
    print "Difference between H_ex1 and H_ex2: max(abs(H_ex1-H_ex2))=%g" % diff13
    print "Max value = %g, relative error = %g " % (max(H_ex_1), diff13/max(H_ex_1))
    assert diff13 < 1e-8
    assert diff13/max(H_ex_1)<1e-15
