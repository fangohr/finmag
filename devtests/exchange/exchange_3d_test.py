import dolfin as df
import numpy as np
import math

N = 5
def compute_b(m):
    interval = df.UnitInterval(N-1)
    V = df.VectorFunctionSpace(interval, "Lagrange", 1, dim=3)

    M    = df.Function(V)
    M.vector()[:] = np.array(m)

    H_ex = df.TrialFunction(V)
    v    = df.TestFunction(V)

    a = df.inner(H_ex, v) * df.dx
    A = df.assemble(a)

    L = - df.inner(df.grad(M), df.grad(v)) * df.dx
    b = df.assemble(L)

    H_ex = df.Function(V)
    df.solve(A, H_ex.vector(), b)
    return H_ex.vector().array()

def rotate(v, a):
    return np.array([
        v[0]*math.cos(a) + v[1]*math.sin(a),
        -v[0]*math.sin(a) + v[1]*math.cos(a),
        0
    ])

for m in np.eye(N*3):
    #print compute_b(m)
    pass

def cross(a, b):
    assert a.shape[0] == 3
    assert a.shape == b.shape
    res = np.empty(a.shape)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = a[2]*b[0] - a[0]*b[2]
    res[2] = a[0]*b[1] - a[1]*b[0]
    return res

m = np.zeros(N*3)
for i in xrange(N):
    m[i::N] = rotate([1,0,0],math.pi*i/(N-1))

b = compute_b(m)

m.shape = (3, -1)
b.shape = (3, -1)

print np.round(cross(m, b),decimals=2)
