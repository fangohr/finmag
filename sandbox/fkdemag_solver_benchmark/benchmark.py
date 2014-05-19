import dolfin as df
import numpy as np
import finmag.energies.demag.fk_demag as fk
from finmag.util.helpers import vector_valued_function
from finmag.native.llg import compute_bem_fk
from aeon import default_timer


def prepare_demag(S3, m, Ms, unit_length, s_param, solver, p_param, preconditioner, bem, boundary_to_global):
    demag = fk.FKDemag()
    if bem is not None:
        # Use the precomputed BEM if it was provided.
        demag.precomputed_bem(bem, boundary_to_global)
    demag.parameters[s_param] = solver
    demag.parameters[p_param] = preconditioner
    demag.setup(S3, m, Ms, unit_length)
    return demag


def prepare_benchmark(S3, m, Ms, unit_length, H_expected=None, tol=1e-3, repeats=10, bem=None, boundary_to_global=None):
    def benchmark(timed_method_name, s_param, solvers, p_param, preconditioners, name):
        results = np.ma.zeros((len(preconditioners), len(solvers)))

        log = open(name + ".log", "w")
        log.write("Benchmark with name {} for params {}, {}.\n".format(name, s_param, p_param))
        for i, prec in enumerate(preconditioners):
            print ""
            print "{:>2}".format(i),
            log.write("\nUsing preconditioner {}.\n".format(prec))
            for j, solv in enumerate(solvers):
                log.write("Using solver {}.\n".format(solv))
                demag = prepare_demag(S3, m, Ms, unit_length, s_param, solv, p_param, prec, bem, boundary_to_global)
                try:
                    checked_result = False
                    for _ in xrange(repeats):  # to average out little fluctuations
                        H = demag.compute_field()  # this can fail
                        if not checked_result:
                            checked_result = True
                            max_diff = np.max(np.abs(H - H_expected))
                            if max_diff > tol:
                                print "x",
                                log.write("Error {:.3} higher than allowed {:.3}.\n".format(max_diff, tol))
                                results[i, j] = np.ma.masked
                                break
                            else:
                                log.write("Result okay.\n")
                                print "o",
                except RuntimeError as e:
                    log.write("Failed with RuntimeError.\n")
                    print "x",
                    results[i, j] = np.ma.masked
                else:
                    time = fk.fk_timer.time_per_call(timed_method_name, "FKDemag")
                    results[i, j] = time
                del(demag)
                default_timer.reset()
                fk.fk_timer.reset()
        log.close()
        results.dump(name + ".pickle")
        return results
    return benchmark


def run_demag_benchmark(m, mesh, unit_length, tol, repetitions=10, name="bench", H_expected=None):
    S3 = df.VectorFunctionSpace(mesh, "CG", 1)
    m = vector_valued_function(m, S3)
    Ms = 1

    # pre-compute BEM to save time
    bem, boundary_to_global = compute_bem_fk(df.BoundaryMesh(mesh, 'exterior', False))

    if H_expected is not None:
        H = vector_valued_function(H_expected, S3)
        H_expected = H.vector().array()
    else: # if no H_expected was passed, use default/default as reference
        demag = fk.FKDemag()
        demag.precomputed_bem(bem, boundary_to_global)
        demag.setup(S3, m, Ms, unit_length)
        H_expected = demag.compute_field()
        del(demag)

    # gather all solvers and preconditioners
    solvers = [s[0] for s in df.krylov_solver_methods()]
    preconditioners = [p[0] for p in df.krylov_solver_preconditioners()]

    benchmark = prepare_benchmark(S3, m, Ms, unit_length, H_expected, tol, repetitions, bem, boundary_to_global)
    results_1 = benchmark("first linear solve",
                          "phi_1_solver", solvers,
                          "phi_1_preconditioner", preconditioners,
                          name=name + "_1")
    results_2 = benchmark("second linear solve",
                          "phi_2_solver", solvers,
                          "phi_2_preconditioner", preconditioners,
                          name=name + "_2")
    return solvers, preconditioners, results_1, results_2
