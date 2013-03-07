import dolfin as df
import numpy as np
from finmag.sim.llg import LLG

def test_spatially_varying_alpha():
    length = 20e-9 # m
    simplices = 10
    mesh = df.IntervalMesh(simplices, 0, length)

    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    llg = LLG(S1, S3)
    
    # one value of alpha on all nodes
    llg.alpha = 0.5 # API hasn't changed.
    expected_alphas = 0.5 * np.ones(simplices+1)
    assert np.array_equal(llg.alpha_vec, expected_alphas) 

    # spatially varying alpha
    multiplicator = df.Function(llg.S1)
    multiplicator.vector()[:] = np.linspace(0.5, 1.5, simplices+1)
    llg.spatially_varying_alpha(0.4, multiplicator) # This is new.
    expected_alphas = np.linspace(0.2, 0.6, simplices+1)
    print "Got:\n", llg.alpha_vec
    print "Expected:\n", expected_alphas
    assert np.allclose(llg.alpha_vec, expected_alphas) 


