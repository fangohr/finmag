import numpy as np
import unittest
import pytest  # D33: added for the not_ported/xfail marker below (master had no pytest import)
try:  # master: from finmag.native import llg as native_llg
    from finmag.native import llg as native_llg
except ImportError:
    native_llg = None  # not ported (D33): tests below xfail
from .test_jacobean_computation import setup_llg_params_near_one
from finmag.util.time_counter import counter


class NativeLlgTests(unittest.TestCase):

    @pytest.mark.not_ported
    @pytest.mark.xfail(reason="not ported: native legacy LLG kernel, NumPy LLG path is current but this native assertion is unmapped (D33)", strict=True)
    def test_llg_performance(self):
        # Unfortunately 100000 nodes is not enough to even fill the L3 cache
        # TODO: Increase the number of nodes when this is fast enough
        llg, m = setup_llg_params_near_one(node_count=200000)
        # Calculate H_eff etc
        llg.solve_for(m, 0)
        # Profile the LLG computation using pyinstant
        c = counter()
        H_eff = llg.effective_field.H_eff.reshape((3, -1))
        m.shape = (3, -1)
        dmdt = np.zeros(m.shape)
        while c.next():
            native_llg.calc_llg_dmdt(m, H_eff, 0.0, dmdt, llg.pins, llg.gamma, llg.alpha.vector(
            ).get_local(), 0.1 / llg.c, llg.do_precession)
        print("Computing dm/dt via native C++ code", c)

if __name__ == "__main__":
    unittest.main()
