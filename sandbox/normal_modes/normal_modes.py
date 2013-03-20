import dolfin as df
from hgviewlib.hggraph import diff
from sympy.mpmath.function_docs import ei
from finmag import Simulation
from finmag.energies import Demag, Exchange, Zeeman
from finmag.util.consts import Oersted_to_SI
import h5py
import scipy.sparse.linalg
import numpy as np
from finmag.util.helpers import fnormalise

SX, SY, SZ = 116, 60, 20

def groundstate_filename(ns):
    return "groundstate-%d-%d-%d" % tuple(ns)

def eigenvector_filename(ns, n):
    return "eigenvector-%d-%d-%d-%d" % (ns[0], ns[1], ns[2], n)

def read_relaxed_state(ns):
    fn = groundstate_filename(ns) + ".h5"
    print "Reading the m vector from", fn
    f = h5py.File(fn, "r")
    return f['/VisualisationVector/0'][...]

def setup_sim(ns, m0):
    nx, ny, nz = ns
    # Fe, ref: PRB 69, 174428 (2004)
    # A = 2.5e-6 erg/cm^3
    # M_s = 1700 emu/cm^3
    # gamma = 2.93 GHz/kOe
    Ms = 1700e3
    A = 2.5e-6*1e-5
    gamma_wrong = 2.93*1e6/Oersted_to_SI(1.) # wrong by a factor of 6 (?)
    Hzeeman = [10e3*Oersted_to_SI(1.), 0, 0]

    mesh = df.BoxMesh(0, 0, 0, SX, SY, SZ, nx, ny, nz)

    sim = Simulation(mesh, Ms)
    sim.set_m(m0)
    sim.add(Demag())
    sim.add(Exchange(A))
    sim.add(Zeeman(Hzeeman))

    return sim


def find_relaxed_state(ns):
    sim = setup_sim(ns, (1, 0, 0))

    print "Finding the relaxed state for", sim.mesh

    def print_progress(sim):
        print "Reached simulation time: {} ns".format(sim.t*1e9)

    sim.schedule(print_progress, every=1e-9)
    sim.relax()

    m = sim.llg._m

    # Save the result
    filename = groundstate_filename(ns) + ".xdmf"
    f = df.File(filename)
    f << m
    f = None
    print "Relaxed field saved to", filename

def differentiate_fd(f, x, dx):
    h = 0.01*np.sqrt(np.dot(x, x))/np.sqrt(np.dot(dx, dx)+1e-100)
    res = np.zeros(dx.size)
    for w, a in zip([1./12., -2./3., 2./3., -1./12.], [-2., -1., 1., 2.]):
        res += (w/h)*f(x + a*h*dx)
    return res

if __name__=="__main__":
    ns = [29, 15, 2]
    m0 = read_relaxed_state(ns)
    sim = setup_sim(ns, m0)
    m0 = sim.m

    n = sim.m.size

    steps = [0]

    def compute_H(m):
        sim.llg._m.vector()[:] = fnormalise(m)
        return sim.llg.effective_field.compute()

    def J_times_vec(dm):
        steps[0] += 1
        return differentiate_fd(compute_H, m0, dm)

    # Solve the eigenvalue problem using ARPACK
    # The eigenvalue problem is not formulated correctly at all
    # The correct formulation is in the paper from d'Aquino
    J = scipy.sparse.linalg.LinearOperator((n,n), matvec=J_times_vec)
    n_values = 3
    w, v = scipy.sparse.linalg.eigs(J, n_values, which='LM')

    print w.shape, v.shape
    print "Computed %d largest eigenvectors for %s" % (n_values,sim.mesh)
    print "Eigenvalues:", w

    for i, x in enumerate(v.T):
        M = df.Function(sim.S3)
#        print x.shape, np.real(x).shape
        M.vector()[:] = np.real(x).copy()
        f = df.File(eigenvector_filename(ns, i+1) + ".xdmf")
        f << M
        f = None


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
