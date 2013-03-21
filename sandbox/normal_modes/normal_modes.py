import dolfin as df
import sys
from finmag import Simulation
from finmag.energies import Demag, Exchange, Zeeman
from finmag.util.consts import Oersted_to_SI
import h5py
import scipy.sparse.linalg
import numpy as np
from finmag.util.helpers import fnormalise

def read_relaxed_state(problem):
    fn = problem.name() + "-groundstate.h5"
    print "Reading the m vector from", fn
    f = h5py.File(fn, "r")
    return f['/VisualisationVector/0'][...]

def find_relaxed_state(problem):
    sim = problem.setup_sim(problem.initial_m())

    print "Finding the relaxed state for ", problem.name(), ", mesh",sim.mesh

    def print_progress(sim):
        print "Reached simulation time: {} ns".format(sim.t*1e9)

    sim.schedule(print_progress, every=1e-9)
    sim.relax()

    m = sim.llg._m

    # Save the result
    filename = sim.name() + "-groundstate.xdmf"
    f = df.File(filename)
    f << m
    f = None
    print "Relaxed field saved to", filename

def differentiate_fd4(f, x, dx):
    h = 0.01*np.sqrt(np.dot(x, x))/np.sqrt(np.dot(dx, dx)+1e-50)
    if isinstance(dx, np.ndarray):
        res = np.zeros(dx.size, dtype=dx.dtype)
    else:
        res = 0.
    for w, a in zip([1./12., -2./3., 2./3., -1./12.], [-2., -1., 1., 2.]):
        res += (w/h)*f(x + a*h*dx)
    return res


def differentiate_fd2(f, x, dx):
    h = 0.01*np.sqrt(np.dot(x, x))/np.sqrt(np.dot(dx, dx)+1e-50)
    if isinstance(dx, np.ndarray):
        res = np.zeros(dx.size, dtype=dx.dtype)
    else:
        res = 0.
    for w, a in zip([-1./2., 1./2.], [-1., 1.]):
        res += (w/h)*f(x + a*h*dx)
    return res

def compute_H_func(sim):
    def compute_H(m):
        # no normalisation since we want linearity
        sim.llg._m.vector()[:] = m
        return sim.llg.effective_field.compute()

    def compute_H_complex(m):
        if np.iscomplexobj(m):
            H_real = compute_H(np.real(m).copy())
            H_imag = compute_H(np.imag(m).copy())
            return H_real + 1j* H_imag
        else:
            return compute_H(m)
    return compute_H_complex

def normalise(m):
    assert m.shape == (3, 1, m.shape[2])
    return m/np.sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2])

def transpose(a):
    return np.transpose(a, [1,0,2])

# Matrix-vector or Matrix-matrix product
def mult(a, b):
    # a and b are ?x?xn arrays where ? = 1..3
    assert len(a.shape) == 3
    assert len(b.shape) == 3
    assert a.shape[2] == b.shape[2]
    assert a.shape[1] == b.shape[0]
    assert a.shape[0] <= 3 and a.shape[1] <= 3
    assert b.shape[0] <= 3 and b.shape[1] <= 3

    # One of the arrays might be complex, so we need to determine the type
    # of the resulting array
    res = np.zeros((a.shape[0], b.shape[1], a.shape[2]), dtype=type(a[0,0,0]+b[0,0,0]))
    for i in xrange(res.shape[0]):
        for j in xrange(res.shape[1]):
            for k in xrange(a.shape[1]):
                res[i,j,:] += a[i,k,:]*b[k,j,:]

    return res

def cross(a, b):
    assert a.shape == (3, 1, a.shape[2])
    res = np.empty(a.shape)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = a[2]*b[0] - a[0]*b[2]
    res[2] = a[0]*b[1] - a[1]*b[0]
    return res

def precompute_arrays(m0):
    n = m0.shape[2]
    assert m0.shape == (3,1,n)

    # Start with e_z and compute e_z x m
    m_perp = cross(m0, [0.,0.,-1.])
    m_perp[2] += 1e-100
    m_perp = cross(normalise(m_perp), m0)
    m_perp = normalise(m_perp)

    # (108)
    R = np.zeros((3,3,n))
    R[:,2,:] = m0[:,0,:]
    R[:,1,:] = m_perp[:,0,:]
    R[:,0,:] = cross(m_perp, m0)[:,0,:]

    # Matrix for the cross product m0 x v
    Mcross = np.zeros((3,3,n))
    Mcross[:,0,:] = cross(m0, [1.,0.,0.])[:,0,:]
    Mcross[:,1,:] = cross(m0, [0.,1.,0.])[:,0,:]
    Mcross[:,2,:] = cross(m0, [0.,0.,1.])[:,0,:]

    B0 = -1j*Mcross
    # (114), multiplying on the left again
    B0p = mult(transpose(R), mult(B0, R))

    # Matrix for the projection onto the plane perpendicular to m0
    Pm0 = np.zeros((3,3,n))
    Pm0[0,0,:] = 1.
    Pm0[1,1,:] = 1.
    Pm0[2,2,:] = 1.
    Pm0 -= mult(m0, transpose(m0))

    # Matrix for the injection from 2n to 3n
    S = np.zeros((3,2,n))
    S[0,0,:] = 1.
    S[1,1,:] = 1.
    # Matrix for the projection from 3n to 2n is transpose(S)

    B0pp = mult(transpose(S), mult(B0p, S))

    # The eigenproblem is
    # D phi = omega phi
    # D = B0pp * S- * R^t (C+H0) R * S+
    # D = Dleft * (C+H0) * Dright
    #
    Dright = mult(R, S).copy()
    Dleft = mult(B0pp, mult(transpose(S), transpose(R))).copy()
    return R, Mcross, Pm0, B0pp, Dleft, Dright

def find_normal_modes(sim):
#    sim = problem.setup_sim(read_relaxed_state(problem))
    m0_flat = sim.m.copy()
    m0 = m0_flat.view()
    m0.shape = (3,1,-1)
    n = m0.shape[2]

    R, Mcross, Pm0, B0pp, Dleft, Dright = precompute_arrays(m0)
    # Compute h0
    compute_H = compute_H_func(sim)
    H0 = compute_H(m0_flat).copy()
    H0.shape = (1,3,n)
    H0 = mult(H0, m0)
    H0.shape = (n,)

    steps = [0]

    def D_times_vec(phi):
        steps[0] += 1
        if steps[0] == 1:
            sys.stderr.write("1\n")
        phi = phi.view()
        # Multiply by Dright
        phi.shape = (2,1,n)
        dm = mult(Dright, phi)
        # Multiply by C+H0
        dm.shape = (-1,)
        if steps[0] == 1:
            sys.stderr.write("21\n")
        v = differentiate_fd4(compute_H, m0_flat, dm)
        if steps[0] == 1:
            sys.stderr.write("31\n")
        v.shape = (3,n)
        dm.shape = (3,n)
        v[0] += H0*dm[0]
        v[1] += H0*dm[1]
        v[2] += H0*dm[2]
        # Multiply by Dleft
        v.shape = (3, 1, n)
        res = mult(Dleft, v)
        res.shape = (-1,)

        if steps[0] % 10 == 0:
            print "Step #", steps[0]

        return res

    # Solve the eigenvalue problem using ARPACK
    # The eigenvalue problem is not formulated correctly at all
    # The correct formulation is in the paper from d'Aquino
    D = scipy.sparse.linalg.LinearOperator((2*n, 2*n), matvec=D_times_vec, dtype=complex)
    n_values = 3
    w, v = scipy.sparse.linalg.eigs(D, n_values, which='LM')

    print w.shape, v.shape
    print "Computed %d largest eigenvectors for %s" % (n_values, sim.mesh)
    print "Eigenvalues:", w

if __name__=="__main__":
    pass
#    df.plot(H_eff)
#    df.interactive()
#
#    external_field = Zeeman((0, Ms, 0))
#    sim.add(external_field)
#    sim.relax()
#    t0 = sim.t # time needed for first relaxation
#
#    external_field.set_value((0, 0, Ms))
#    sim.relax()
#    t1 = sim.t - t0 # time needed for second relaxation
#
#    assert sim.t > t0
#    assert abs(t1 - t0) < 1e-10
