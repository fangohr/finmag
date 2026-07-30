import numpy as np
import pytest  # D33: added for the not_ported/xfail marker below (master had no pytest import)
try:  # master: from finmag.native.llg import StochasticHeunIntegrator
    from finmag.native.llg import StochasticHeunIntegrator
except ImportError:
    StochasticHeunIntegrator = None  # not ported (D33): tests below xfail


@pytest.mark.not_ported
@pytest.mark.xfail(reason="not ported: legacy Heun driver (register M16)", strict=True)
def test_file_builds():
    drift = lambda y, t: 2 * y
    diffusion = lambda y, t: y + 0.1
    integrator = StochasticHeunIntegrator(np.zeros(1), drift, diffusion, 1e-12)
    integrator.helloWorld()
