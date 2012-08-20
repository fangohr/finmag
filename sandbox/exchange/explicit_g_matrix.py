import dolfin as df

def test_computing_exchange_matrix_g():
    nm = 1e-9
    nx = ny = 4
    mesh = df.Rectangle(0,0,20*nm,20*nm,nx,ny)
    print "mesh.shape:",mesh.coordinates().shape
    V = df.VectorFunctionSpace(mesh,"CG",1,dim=3)
    #v = df.TestFunction(V)

    exchange_factor = 1
    Minitial = df.Constant([1,0.2,0.1])
    M = df.interpolate(Minitial,V)
    Mvec = M.vector().array()
    Mvec[15]=0.5 #change some entry to force non-zero exchange field
    M.vector()[:]=Mvec
    E = exchange_factor * df.inner(df.grad(M), df.grad(M)) * df.dx
    #dE_dM = df.derivative(E, M, v)
    dE_dM = df.derivative(E, M)
    H_ex = df.assemble(dE_dM).array()
    print "H_ex:",H_ex.shape
    g_form = df.derivative(dE_dM,M)
    g = df.assemble(g_form).array()

    print "g.shape: ",g.shape
    import numpy as np
    Mvec = M.vector().array()
    print "Mvec.shape:",Mvec.shape
    H_ex2 = np.dot(g,Mvec)
    print "H_ex2:",H_ex2.shape
    maxdiff = max(abs(H_ex2-H_ex))
    print "Difference between two ways of computing H_ex:", (maxdiff)
    assert maxdiff < 1e-14

    print "Maximum field value: H_ex:%g H_ex2:%g " % (max(H_ex),max(H_ex2))

    print "H_ex:",np.round(H_ex,2)

    

def box_assemble_time_over_box_matrix_time(n):
    """Simulation 1 is computing H_ex=dE_dM via assemble.
    Simulation 2 is computing H_ex=g*M with a suitable pre-computed matrix g.
    
    Here we look briefly at the performance that the two methods provide.

    The way this is currently implemented (with numpy arrays to store
    sparse (!) matrix g), the box-matrix method is faster for small n 
    (around n=15); presumably due to the size of the g-matrix which is stored
    in full when represented as numpy array.

    Need to switch to dolfin's sparse matrix represenation using, for
    example, petsc as the backend, and do the multiplication g*M in petsc,
    before converting to numpy.array.

    """

    import time
    import finmag.sim.exchange as exchange

    m_initial = (
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0')

    nm = 1e-9
    nx = ny = n
    length = 20*nm
    mesh = df.Rectangle(0,0,length,length,nx,ny)
    print "mesh.shape:",mesh.coordinates().shape
    V = df.VectorFunctionSpace(mesh,"CG",1,dim=3)
    import time
    C = 1.3e-11 # J/m exchange constant
    m = df.interpolate(df.Expression(m_initial,L=length),V)

    Ms = 0.86e6#A/m, for example Py
    exchange_object1 = exchange.Exchange(V,m,C,Ms, method='box-assemble')
    exchange_object2 = exchange.Exchange(V,m,C,Ms, method='box-matrix-numpy') 
    
    Ntest = 10
    print "Calling exchange calculation %d times (%d box assemblies)" % (Ntest,Ntest)
    time_start = time.time()
    for i in xrange(Ntest):
        H_ex1 = exchange_object1.compute_field()
    time_end = time.time()
    time1 = time_end-time_start
    print "Box-assemble took %g seconds" % time1

    print "Calling exchange calculation %d times (%d matrix * vector operations)" % (Ntest,Ntest)
    time_start = time.time()
    for i in xrange(Ntest):
        H_ex2 = exchange_object2.compute_field()
    time_end = time.time()
    time2 = time_end-time_start
    print "Box-matrix took %g seconds" % time2
    print "Speed up: matrix/assembly-method = %g" % (time1/time2)

    
    diff = max(abs(H_ex1-H_ex2))
    print "Difference between H_ex1 and H_ex2: max(abs(H_ex1-H_ex2))=%g" % diff
    print "Max value = %g, relative error = %g " % (max(H_ex1), diff/max(H_ex1))
    assert diff < 10e-8
    assert diff/max(H_ex1)<1e-15

    return time1/time2


if __name__ == "__main__":
    print "Testing correctness"
    test_computing_exchange_matrix_g()

    ns = [1,5,10,15,20]
    speedup = []
    for n in ns:
        speedup.append(box_assemble_time_over_box_matrix_time(n))

    print "Speed up (larger than 1 means matrix method is faster than assembly"
    for n,s in zip(ns,speedup):
        print "%3d -> %5.2f" % (n,s)
