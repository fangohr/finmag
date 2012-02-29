import dolfin as df
import time
from finmag.sim.exchange import Exchange as exch
import numpy as np

def efficiency_test(method, n):
    """
    Test the efficiency of different methods
    implemented in the Exchange class.

    """

    m0 = ("(2*x[0]-L)/L",
          "sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))",
          "0.0")

    length = 20e-9
    mesh = df.Rectangle(0, 0, length, length, n, n)
    print "mesh.shape:",mesh.coordinates().shape

    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)

    C = 1.3e-11 # J/m exchange constant

    m = df.interpolate(df.Expression(m0,L=length), V)

    Ms = 0.86e6 # A/m, for example Py
    exchange_object = exch(V, m, C, Ms, method=method)

    Ntest = 10
    print "Calling exchange calculation %d times." % Ntest
    time_start = time.time()
    for i in xrange(Ntest):
        H_ex = exchange_object.compute_field()
    time_end = time.time()
    t1 = time_end-time_start
    print "Method '%s' took %g seconds" % (method, t1)

    return H_ex, t1

def correctness_test(results, ref_method, tol, rtol):
    """
    Test correctness of the different methods by comparing
    their results with eachother.
    
    Worst case scenario: Everyone is equal to eachother, but 
    they are all wrong. Then we have a problem...

    """
    
    # Sanity check
    if not results.has_key(ref_method):
        print "Cannot use %s method as reference." % ref_method
        return

    ref = results[ref_method]

    for method in results.iterkeys():
        if method == ref_method:
            continue
        Hex = results[method]
        assert len(ref) == len(Hex)

        max_error, rel_error = 0, 0

        for i in range(len(ref)):
            diff = abs(ref[i] - Hex[i])
            max_error2 = diff.max()
            rel_error2 = max_error2/max(ref[i])
            
            if max_error2 > max_error:
                max_error = max_error2
            if rel_error2 > rel_error:
                rel_error = rel_error2

        print "Max difference between '%s' and '%s' methods:" % \
                (ref_method, method), max_error

        print "Relative error:", rel_error
        
        assert max_error < tol
        assert rel_error < rtol

def print_results(results, ns):
    """
    Print timings and speedups.

    """

    print "\n\n*** Timings ***"
    for i, n in enumerate(ns):
        print "\nn = %d" % n
        for method in results.iterkeys():
            print "'%s': %.2f ms" % (method, results[method][i]*1000)

    print "\n*** Speedup ***"
    print "(Larger than 1 means second method is faster than first method.)"
    
    methods = results.keys()
    nomethods = len(methods)

    for i in range(nomethods - 1):
        for j in range(i + 1, nomethods):
            print "\n** '%s' vs '%s' **" % (methods[i], methods[j])
            for k, n in enumerate(ns):
                t1 = results[methods[i]][k]
                t2 = results[methods[j]][k]
                sp = t1/t2
                print "%3d -> %5.2f" % (n, sp)


if __name__ == '__main__':
    methods = ['box-assemble', 'box-matrix-numpy', 'box-matrix-petsc']#, 'project']
    ns = [1, 5, 10, 20, 50]

    methods_times = {}
    methods_field = {}
    for method in methods:
        times = []
        field = []
        for n in ns:
            H_ex, t = efficiency_test(method, n)
            field.append(H_ex)
            times.append(t)
        methods_times[method] = times
        methods_field[method] = field
    correctness_test(methods_field, 'box-matrix-numpy', 10e-7, 1e-14)
    print_results(methods_times, ns)
    
