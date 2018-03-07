import dolfin as df
import math
import numpy as np
from finmag import Simulation
from finmag.energies import Demag, Zeeman, Exchange
from finmag.util.consts import mu0, Oersted_to_SI
from finmag.util.helpers import fnormalise
from normal_modes import differentiate_fd4
from normal_modes import differentiate_fd2
from normal_modes import compute_H_func, normalise, mult, transpose, precompute_arrays, cross, find_normal_modes, compute_A

def test_differentiate_fd4():
    def f(x):
        return np.array([math.exp(x[0])+math.sin(x[1]), 0.])

    x = np.array([1.2, 2.4])
    dx = np.array([3.5, 4.5])
    value = dx[0]*math.exp(x[0]) + dx[1]*math.cos(x[1])
    assert abs(differentiate_fd4(f, x, dx)[0] - value) < 1e-8

def test_differentiate_quadratic():
    def f(x):
        return x**2 + 3.5*x + 7

    assert abs(differentiate_fd4(f, 2, 3) - (2*2+3.5)*3) < 1e-12
    assert abs(differentiate_fd2(f, 2, 3) - (2*2+3.5)*3) < 1e-12

def test_differentiate_heff():
    ns = [2, 2, 2]
    mesh = df.BoxMesh(0, 0, 0, 1, 1, 1, *ns)
    sim = Simulation(mesh, 1.2)
    sim.set_m([1,0,0])
    sim.add(Demag())
    sim.add(Exchange(2.3*mu0))

    compute_H = compute_H_func(sim)

    # Check that H_eff is linear without Zeeman
    np.random.seed(1)
    m1 = fnormalise(np.random.randn(*sim.m.shape))
    m2 = fnormalise(np.random.randn(*sim.m.shape))
    # TODO: need to use a non-iterative solver here to increase accuracy
    assert np.max(np.abs(compute_H(m1)+compute_H(m2) - compute_H(m1+m2))) < 1e-6
    # Add the zeeman field now
    sim.add(Zeeman([2.5,3.5,4.3]))

    # Check that both fd2 and fd4 give the same result
    assert np.max(np.abs(differentiate_fd4(compute_H, m1, m2)-differentiate_fd2(compute_H, m1, m2))) < 1e-10

def norm(a):
    a = a.view()
    a.shape = (-1,)
    a = np.abs(a) # for the complex case
    return np.sqrt(np.dot(a,a))

def test_compute_arrays():
    n = 1
    m0 = normalise(np.array([1.,2.,3.]).reshape((3,1,n)))

    R, Mcross, Pm0, B0pp, Dleft, Dright = precompute_arrays(m0)
    # R is a rotation matrix
    assert norm(mult(transpose(R), R)[:,:,0]-np.eye(3)) < 1e-14
    assert np.abs(np.linalg.det(R[:,:,0]) - 1.) < 1e-14
    # Mcross is the matrix for the cross product m0 x v
    assert norm(np.dot(Mcross[:,:,0], [0.,0.,1.]) - cross(m0, [0.,0.,1.])[:,0,0]) < 1e-14
    # B0pp is a hermitian matrix with pure imaginary entries
    assert norm(mult(B0pp, B0pp)[:,:,0] - np.diag([1,1,])) < 1e-14
    # Pm0: Matrix for the projection onto the plane perpendicular to m0
    assert norm(mult(Pm0, m0)) < 1e-14


def test_compute_main():
    Ms = 1700e3
    A = 2.5e-6*1e-5
    Hz = [10e3*Oersted_to_SI(1.), 0, 0]

    mesh = df.BoxMesh(0, 0, 0, 50, 50, 10, 6, 6, 2)
    mesh = df.BoxMesh(0, 0, 0, 5, 5, 5, 1, 1, 1)
    print mesh

    # Find the ground state
    sim = Simulation(mesh, Ms)
    sim.set_m([1,0,0])
    sim.add(Demag())
    sim.add(Exchange(A))
    sim.add(Zeeman(Hz))
    sim.relax()

    # Compute the eigenvalues - does not converge
#    find_normal_modes(sim)

def test_compute_A():
    Ms = 1700e3
    A = 2.5e-6 * 1e-5
    Hz = [10e3 * Oersted_to_SI(1.), 0, 0]
    mesh = df.BoxMesh(0, 0, 0, 5, 5, 5, 2, 2, 2)
    sim = Simulation(mesh, Ms)
    sim.set_m([1, 0, 0])
    sim.add(Demag())
#    sim.add(Exchange(A))
#    sim.add(Zeeman(Hz))
    sim.relax()

    print mesh
    A0, m0 = compute_A(sim)
    print A0[:3,:3]
    print m0
