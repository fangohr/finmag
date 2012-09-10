import dolfin as df
import time
from finmag.energies.anisotropy import UniaxialAnisotropy as ani
from finmag.energies.exchange import Exchange as exch
import numpy as np

def efficiency_test(method, n, field='exchange'):
    """
    Test the efficiency of different methods
    implemented in the different energy classes.

    Possible fields so far is 'exchange' and 'anisotropy'.

    """

    length = 20e-9
    mesh = df.Rectangle(0, 0, length, length, n, n)
    print "mesh.shape:",mesh.coordinates().shape
    V = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)

    # Initial magnetisation
    m0 = ("0.0",
          "sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))",
          "(2*x[0]-L)/L")
    m = df.interpolate(df.Expression(m0,L=length), V)
    Ms = 0.86e6 # A/m, for example Py

    if field == 'exchange':
        C = 1.3e-11 # J/m exchange constant
        energy = exch(C, method=method)
        energy.setup(V, m, Ms)

    elif field == 'anisotropy':
        a = df.Constant((0, 0, 1)) # Easy axis
        K = 520e3 # J/m^3, Co
        energy = ani(V, K, a, method=method)
        energy.setup(V, m, Ms)

    else:
        raise NotImplementedError("%s is not implemented." % field)

    Ntest = 10
    print "Calling exchange calculation %d times." % Ntest
    time_start = time.time()
    for i in xrange(Ntest):
        H_ex = energy.compute_field()
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
    print "\n\n\n*** Comparisons ***"

    for method in results.iterkeys():
        if method == ref_method:
            continue
        Hex = results[method]
        assert len(ref) == len(Hex)

        max_error, rel_error = 0, 0

        for i in range(len(ref)):
            diff = abs(ref[i] - Hex[i])
            max_error2 = diff.max()
            rel_error2 = max_error2/max(abs(ref[i]))

            if max_error2 > max_error:
                max_error = max_error2
            if rel_error2 > rel_error:
                rel_error = rel_error2

        print "\nBetween '%s' and '%s' methods:" % \
                (ref_method, method)

        print "Max error:     ", max_error
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
            print "n ="
            for k, n in enumerate(ns):
                t1 = results[methods[i]][k]
                t2 = results[methods[j]][k]
                sp = t1/t2
                print "%6d -> %5.2f" % (n, sp)


if __name__ == '__main__':

    # This is the method to compare all
    # the other methods against. This
    # should be the one we are most sure
    # is working correct.
    ref_method = 'box-matrix-numpy'

    # Field for which to compare efficiency and correctness
    # (Comment out the one you are not interested in)
    #field = "anisotropy"
    field = "exchange"

    methods = ['box-assemble', 'box-matrix-numpy', 'box-matrix-petsc']#, 'project']
    ns = [1, 5, 10, 20, 50]

    methods_times = {}
    methods_field = {}
    for method in methods:
        times  = []
        fields = []
        for n in ns:
            H_ex, t = efficiency_test(method, n, field)
            fields.append(H_ex)
            times.append(t)
        methods_times[method] = times
        methods_field[method] = fields

    correctness_test(methods_field, ref_method, 10e-7, 1e-14)
    print_results(methods_times, ns)
    print "These results were obtained for the %s field." % field

