from dolfin import *
import numpy as np
from sim.anisotropy import Anisotropy

# Mesh and functionspace
m = 1e-8
mesh = Box(0,m,0,m,0,m,1,1,1)
V  = VectorFunctionSpace(mesh, "CG", 1)

# Anisotropy constant
K1 = Constant(520e3) # For Co (J/m^3), according to Nmag (0.2.1) example 2.10

# Initial direction of the magnetic field.
Ms = 1
M0 = Constant((1./np.sqrt(2), 0, 1./np.sqrt(2)))
M  = project(M0, V)

# Easy axes
a = Constant((0, 0, 1))

# Anisotropy gradient
ani = Anisotropy(V, M, Ms, K1, a)
dE_dM = ani.compute()

# Manually derivative
W = TestFunction(V)
g_ani_form = -K1*(2*dot(a, M)*dot(a, W))*dx
vol = assemble(dot(W, Constant((1,1,1)))*dx).array()
man_grad = assemble(g_ani_form).array() / vol

def test_compare_grads():

    global dE_dM
    global man_grad

    l = len(dE_dM)
    assert l == len(man_grad)

    TOL = 1e-5
    diff = dE_dM - man_grad
    for i in range(l):
        assert abs(diff[i]) < TOL

