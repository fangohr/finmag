from dolfin import *

def c():
    return 1e8

def M0():
    Ms = 1e8
    return Ms, Constant((0.8*Ms, -0.1*Ms*2, 0.1*Ms*0))

