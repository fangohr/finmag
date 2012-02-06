import numpy
import dolfin
import pytest
from scipy.integrate import odeint

from ..src.llg import LLG
from ..src.helpers import make_vectors_function,angle

@pytest.mark.xfail
def test_all_orientations():

  length = 20e-9 # m
  simplexes = 10
  mesh = dolfin.Interval(simplexes, 0, length) 

  def angles_after_a_nanosecond(initial_M):
    llg = LLG(mesh)
    llg.initial_M_expr(initial_M, L=length, MS=llg.MS)
    llg.setup()
    llg.pins = [0, 10]

    ts = numpy.linspace(0, 1e-9, 2)
    ys, infodict = odeint(llg.solve_for, llg.M, ts, atol=10, full_output=True)

    vectors = make_vectors_function(ys[0])
    M = vectors(ys[-1])
    angles = [angle(M[i], M[i+1]) for i in xrange(len(M)-1)]
    return angles

  left_right = 'MS * (2*x[0]/L - 1)'
  up_down = 'sqrt(MS*MS - MS*MS*(2*x[0]/L - 1)*(2*x[0]/L - 1))'

  possible_orientations = [
    (left_right, up_down, '0'), (left_right, '0', up_down),
    (up_down, left_right, '0'), (up_down, '0', left_right),
    ('0', left_right, up_down), ('0', up_down, left_right)]

  for M0 in possible_orientations:
    angles = angles_after_a_nanosecond(M0)
    print angles
    assert abs(angles[0] - angles[1]) < 0.05
