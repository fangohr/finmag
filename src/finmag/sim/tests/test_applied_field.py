import dolfin as df
from scipy.integrate import odeint
from finmag.sim.llg import LLG

TOLERANCE = 1e-2

def test_aligns_with_applied_field():
    llg = LLG(df.UnitCube(2, 2, 2))
    llg.initial_M((llg.Ms, 0, 0))
    llg.H_app = (0, 0, llg.Ms/2)
    llg.setup(False)

    ts = [0, 1e-9]
    ys = odeint(llg.solve_for, llg.M, ts, atol=0.1)

    Mx, My, Mz = llg.average_M()

    assert abs(Mx) < TOLERANCE
    assert abs(My) < TOLERANCE
    assert abs(llg.Ms - Mz) < TOLERANCE

