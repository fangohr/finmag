import dolfin as df
from scipy.integrate import odeint
from finmag.sim.llg import LLG

TOLERANCE = 5e-10

def test_aligns_with_applied_field():
    llg = LLG(df.UnitCube(2, 2, 2))
    llg.set_m0((1, 0, 0))
    llg.H_app = (0, 0, llg.Ms/2)
    llg.setup(False)

    ts = [0, 1e-9]
    ys = odeint(llg.solve_for, llg.m, ts)

    mx, my, mz = llg.m_average

    assert abs(mx) < TOLERANCE
    assert abs(my) < TOLERANCE
    assert abs(1 - mz) < TOLERANCE

