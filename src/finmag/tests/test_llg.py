import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.util.helpers import components

def test_method_of_computing_the_average_matters():
    length = 20e-9 # m
    simplices = 10
    mesh = df.IntervalMesh(simplices, 0, length)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)

    llg = LLG(S1, S3)
    llg.set_m((
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0'), L=length)

    average1 = llg.m_average
    average2 = np.mean(components(llg.m), axis=1)
    diff = np.abs(average1 - average2)
    assert diff.max() > 5e-2
