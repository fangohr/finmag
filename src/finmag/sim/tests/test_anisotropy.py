from dolfin import *
import numpy as np
from finmag.sim.anisotropy import Anisotropy

"""
The equation for uniaxial anisotropy reads

.. math::

    E = K_1 V (1 - \gamma^2),

where $V$ is the volume of the domain,
$K_1$ the anisotropy constant, and $\gamma$
the dot product between initial magnetisation
and the easy axis.


Case 1: Parallel -> $a = M = (0, 0, 1)$

The dot product becomes 1, and thus $E$ should be zero.


Case 2: Orthogonal -> $a = (0, 0, 1), M = (0, 1, 0)$

The dot product becomes 0, and the energy should be
equal to $K_1$ multiplied with the volume of the mesh.


Case 3: Compare anisotropy gradient with manually derived

"""

# Tolerance
TOL = 1e-20

# Mesh and functionspace
m = 1e-8
n = 5
mesh = Box(0,m,0,m,0,m,n,n,n)
V  = VectorFunctionSpace(mesh, "CG", 1)

K  = 520e3  # For Co (J/m^3), according to Nmag (0.2.1) example 2.10
K1 = Constant(K) 
Ms = 1

# Easy axes
a = Constant((0, 0, 1))

def test_parallel():
    # Case 1
    M = project(Constant((0, 0, 1)), V)
    energy = Anisotropy(V, M, Ms, K1, a).energy()
    assert abs(energy) < TOL

def test_orthogonal():
    # Case 2
    M = project(Constant((0,1,0)), V)   
    energy = Anisotropy(V, M, Ms, K1, a).energy()
    vol = K*assemble(Constant(1)*dx, mesh=mesh)
    assert abs(energy - vol) < TOL

def test_gradient():
    # Case 3
    M = project(Constant((1./np.sqrt(2), 0, 1./np.sqrt(2))), V)
    dE_dM = Anisotropy(V, M, Ms, K1, a).compute()
    
    # Manually derivative
    w = TestFunction(V)
    g_ani = -K1*(2*dot(a, M)*dot(a, w))*dx
    vol = assemble(dot(w, Constant((1, 1, 1)))*dx).array()
    man_grad = assemble(g_ani).array() / vol
    
    l = len(dE_dM)
    assert l == len(man_grad)

    diff = dE_dM - man_grad
    for i in range(l):
        assert abs(diff[i]) < TOL

