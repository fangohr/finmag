from dolfin import *
import numpy as np
from finmag.sim.anisotropy import UniaxialAnisotropy

"""
The equation for uniaxial anisotropy reads

.. math::

    E = K_1 V (1 - \gamma^2),

and 

.. math::

   \gamma = \angle(\vec{a},\vec{m})

where $V$ is the volume of the domain,
$K_1$ the anisotropy constant, and $\gamma$
the dot product between initial magnetisation
and the easy axis a.


Case 1: Parallel -> $a = M = (0, 0, 1)$

The dot product becomes 1, and thus $E$ should be zero.


Case 2: Orthogonal -> $a = (0, 0, 1), M = (0, 1, 0)$

The dot product becomes 0, and the energy should be
equal to $K_1$ multiplied with the volume of the mesh.


Case 3: Compare anisotropy gradient with manually derived

"""

# Tolerance
TOL = 1e-14

# Mesh and functionspace
L = 10e-9    #10 nm
n = 5
mesh = Box(0,0,0,L,L,L,n,n,n)
V  = VectorFunctionSpace(mesh, "CG", 1)

K  = 520e3  # For Co (J/m^3), according to Nmag (0.2.1) example 2.10
#K1 = Constant(K) 
Ms = 1

# Easy axes
a = Constant((0, 0, 1))

def test_parallel():
    # Case 1
    M = interpolate(Constant((0, 0, 1)), V)
    energy = UniaxialAnisotropy(V, M, K, a).compute_energy()
    print 'test_parallel()'
    print 'This sould be zero %g:' % energy

    energy_scale = -K*assemble(Constant(1)*dx, mesh=mesh) #energy if a and m perpendicular
    print 'Energy difference %g:' % (energy-energy_scale)
    print 'Relative energy difference %g:' % ((energy-energy_scale)/energy_scale)
    assert abs(energy-energy_scale)/energy_scale < TOL

def test_orthogonal():
    # Case 2
    print "test_orthogonal:"
    M = interpolate(Constant((0,1,0)), V)   
    energy = UniaxialAnisotropy(V, M, K, a).compute_energy()
    energy_direct = K*assemble(Constant(0)*dx, mesh=mesh)
    print 'These should be equal: %g-%g=%g ' %( energy, energy_direct, energy-energy_direct)
    assert abs(energy - energy_direct) < TOL

def test_gradient():
    # Case 3
    M = interpolate(Constant((1./np.sqrt(2), 0, 1./np.sqrt(2))), V)
    ani = UniaxialAnisotropy(V, M, K, a)
    dE_dM = ani.compute_field()

    # Manually derivative
    w = TestFunction(V)
    g_ani = Constant(K)*(2*dot(a, M)*dot(a, w))*dx
    vol = assemble(dot(w, Constant((1, 1, 1)))*dx).array()
    man_grad = assemble(g_ani).array() / vol
    
    l = len(dE_dM)
    print 'These array lengths should be equal:', l, len(man_grad)
    assert l == len(man_grad)

    diff = dE_dM - man_grad
    maxdiff = max(abs(diff))

    maxfield = max(abs(dE_dM))
    print "Maximum field entry: %g" % maxfield
    print 'This should be close to zero:', maxdiff/maxfield
    assert maxdiff/maxfield < 1e-15

if __name__ == '__main__':
    test_parallel()
    test_orthogonal()
    test_gradient()

