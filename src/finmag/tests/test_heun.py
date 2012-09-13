import numpy as np
from finmag.native.llg import StochasticHeunIntegrator

def test_file_builds():
    drift = lambda y, t: 2 * y
    diffusion = lambda y, t: y + 0.1
    integrator = StochasticHeunIntegrator(np.zeros(1), drift, diffusion, 1e-12)
    integrator.helloWorld()
