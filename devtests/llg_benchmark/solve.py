import cProfile
import numpy as np
from finmag.native import llg as native_llg
from finmag.util.helpers import fnormalise

dims = 3
nodes = 1e6

m = fnormalise(np.random.random_sample(dims * nodes))
H = fnormalise(np.random.random_sample(dims * nodes))

t = 0.0
pins = np.zeros(0, dtype="int") 
gamma =  2.210173e5
alpha = np.ones(nodes) * 0.5
char_time = 1e-12
do_damping    = True
do_relaxation = True
do_precession = True

def solve():
    dmdt = np.zeros((dims, nodes))

    def python_wrapper_for_c_fun():
        """
        Calls to calc_llg_dmdt won't be recorded by the cProfiler, so we need
        to wrap them in Python code.

        """
        native_llg.calc_llg_dmdt(_m, _H, t, dmdt, pins, gamma, alpha, char_time, do_precession)
        #native_llg.new_llg_dmdt(_m, _H, t, dmdt, pins, gamma, alpha, char_time, do_damping, do_relaxation, do_precession)
        #native_llg.new_llg_dmdt2(_m, _H, t, dmdt, pins, gamma, alpha, char_time, 0b11100) # bit mask

    for i in xrange(1000):
        # This is pretty close to sim/llg.py LLG.solve()
        # except it doesn't compute the effective field.
        _m = m.reshape((3, -1))
        _H = H.reshape((3, -1))
        python_wrapper_for_c_fun()
        dmdt = np.zeros(_m.shape)
        dmdt.reshape((3, -1))

cProfile.run("solve()")
