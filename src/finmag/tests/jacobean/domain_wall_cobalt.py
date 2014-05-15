import numpy as np
import dolfin as df
import math
from finmag.physics.llg import LLG
from finmag.integrators.llg_integrator import llg_integrator
from finmag.energies import Exchange, UniaxialAnisotropy

# Material parameters
Ms_Co = 1400e3 # A/m
K1_Co = 520e3 # A/m
A_Co = 30e-12 # J/m

LENGTH = 100e-9
NODE_COUNT = 100

# Initial m
def initial_m(xi, node_count):
    mz = 1. - 2. * xi / (node_count - 1)
    my = math.sqrt(1 - mz * mz)
    return [0, my, mz]

# Analytical solution for the relaxed mz
def reference_mz(x):
    return math.cos(math.pi / 2 + math.atan(math.sinh((x - LENGTH / 2) / math.sqrt(A_Co / K1_Co))))

def setup_domain_wall_cobalt(node_count=NODE_COUNT, A=A_Co, Ms=Ms_Co, K1=K1_Co, length=LENGTH, do_precession=True):
    mesh = df.IntervalMesh(node_count - 1, 0, length)
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    llg = LLG(S1, S3)
    llg.set_m(np.array([initial_m(xi, node_count) for xi in xrange(node_count)]).T.reshape((-1,)))

    exchange = Exchange(A)
    exchange.setup(S3, llg._m, Ms)
    llg.effective_field.add(exchange)
    anis = UniaxialAnisotropy(K1, (0, 0, 1))
    anis.setup(S3, llg._m, Ms)
    llg.effective_field.add(anis)
    llg.pins = [0, node_count - 1]
    return llg

def domain_wall_error(ys, node_count):
    m = ys.view()
    m.shape = (3, -1)
    return np.max(np.abs(m[2] - [reference_mz(x) for x in np.linspace(0, LENGTH, node_count)]))

def compute_domain_wall_cobalt(end_time=1e-9):
    llg = setup_domain_wall_cobalt()
    integrator = llg_integrator(llg, llg.m)
    integrator.advance_time(end_time)
    return np.linspace(0, LENGTH, NODE_COUNT), llg.m.reshape((3, -1))

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xs, m = compute_domain_wall_cobalt()
    print "max difference between simulation and reference: ", domain_wall_error(m, NODE_COUNT)
    xs = np.linspace(0, LENGTH, NODE_COUNT)
    plt.plot(xs, np.transpose([m[2], [reference_mz(x) for x in xs]]), label=['Simulation', 'Reference'])
    plt.xlabel('x [m]')
    plt.ylabel('m')
    plt.title('Domain wall in Co')
    plt.show()


