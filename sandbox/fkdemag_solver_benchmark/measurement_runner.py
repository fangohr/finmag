import dolfin as df
import pickle
import numpy as np
import finmag.energies.demag.fk_demag as fk
import matplotlib.pyplot as plt
from table_printer import TablePrinter
from finmag.util.helpers import vector_valued_function
from finmag.native.llg import compute_bem_fk
from aeon import default_timer


def _prepare_demag_object(S3, m, Ms, unit_length, s_param, solver, p_param, preconditioner, bem, boundary_to_global):
    demag = fk.FKDemag()
    if bem is not None:
        # Use the precomputed BEM if it was provided.
        demag.precomputed_bem(bem, boundary_to_global)
    demag.parameters[s_param] = solver
    demag.parameters[p_param] = preconditioner
    demag.setup(S3, m, Ms, unit_length)
    return demag


def _loaded_results_table(timed_method_name, results, solvers, preconditioners):
    print "Time shown for {} in s.".format(timed_method_name)
    print "Results loaded from file.\n"

    table = TablePrinter(preconditioners, xlabels_width_min=10)
    for solver in solvers:
        table.new_row(solver)
        for prec in preconditioners:
            try:
                time = results[solver][prec]
            except KeyError:
                time = "-"
            table.new_entry(time)

    default = results['default']['default']
    fastest = fastest_run(results, solvers)

    print "\n\nDefault combination ran in {:.3} s.".format(default)
    print "Fastest combination {}/{} ran in {:.3} s.".format(fastest[0], fastest[1], fastest[2])
    print "That is an {:.1%} improvement.\n".format(1 - fastest[2] / default)


class ToleranceNotReachedException(Exception):
    pass


def create_measurement_runner(S3, m, Ms, unit_length, H_expected=None, tol=1e-3, repeats=10, bem=None, boundary_to_global=None):
    def runner(timed_method_name, s_param, solvers, p_param, preconditioners, skip=[], full_timings_log="timings.txt", results_cache=None):
        r = np.ma.zeros((len(solvers), len(preconditioners)))
        results, failed = {}, []

        try:
            with open(str(results_cache), "r") as f:
                results = pickle.load(f)
                _loaded_results_table(timed_method_name, results, solvers, preconditioners)
                return results, failed
        except IOError:
            print "No recorded results found. Will run benchmarks."
        print "Time shown for {} in s.\n".format(timed_method_name)

        for timer in (default_timer, fk.fk_timer):
            timer.reset()
        log = open(full_timings_log, "w")

        table = TablePrinter(preconditioners)
        for solver in solvers:
            table.new_row(solver)
            #print "\nsolving with {}\n".format(solver)
            results_for_this_solver = {}
            # Compute the demagnetising field with the current solver and each of the preconditioners.
            for prec in preconditioners:
                #print "\nprec {} entry {}\n".format(prec, table.entry_counter)
                if (solver, prec) in skip:
                    table.new_entry("s")
                    continue
                demag = _prepare_demag_object(S3, m, Ms, unit_length, s_param, solver, p_param, prec, bem, boundary_to_global)
                checked_result = False
                try:
                    for _ in xrange(repeats):  # Repeat to average out little fluctuations.
                        H = demag.compute_field()  # This can fail with some method/preconditioner combinations.
                        if not checked_result:
                            # By checking here, we can avoid computing the
                            # field ten times if it's wrong.
                            checked_result = True
                            max_diff = np.max(np.abs(H - H_expected))
                            if max_diff > tol:
                                # We need to raise an exception, otherwise the 'else' clause below is
                                # executed and the measurement will be recorded twice.
                                raise ToleranceNotReachedException
                except ToleranceNotReachedException:
                    table.new_entry("x")
                    failed.append({'solver': solver, 'preconditioner': prec, 'message': "error {:.3} higher than allowed {:.3}"})
                except RuntimeError as e:
                    default_timer.get("compute_field", "FKDemag").stop()
                    table.new_entry("x")
                    failed.append({'solver': solver, 'preconditioner': prec, 'message': e.message})
                else:
                    measured_time = fk.fk_timer.time_per_call(timed_method_name, "FKDemag")
                    table.new_entry(measured_time)
                    results_for_this_solver[prec] = measured_time
                log.write("\nTimings for {} with {}.\n".format(solver, prec))
                log.write(fk.fk_timer.report() + "\n")
                default_timer.reset()
                fk.fk_timer.reset()
                del(demag)
            results[solver] = results_for_this_solver

        if results_cache is not None:
            with open(results_cache, "w") as f:
                pickle.dump(results, f)

        log.close()

        default = results['default']['default']
        fastest = fastest_run(results, solvers)

        print "\nDefault combination ran in {:.3} s.".format(default)
        print "Fastest combination {}/{} ran in {:.3} s.".format(fastest[0], fastest[1], fastest[2])
        print "That is an {:.1%} improvement.".format(1 - fastest[2] / default)

        return results, failed
    return runner


def run_measurements(m, mesh, unit_length, tol, repetitions=10, H_expected=None, name="", skip=[]):
    S3 = df.VectorFunctionSpace(mesh, "CG", 1)
    m = vector_valued_function(m, S3)
    Ms = 1
    bem, boundary_to_global = compute_bem_fk(df.BoundaryMesh(mesh, 'exterior', False))

    if H_expected is not None:
        H = vector_valued_function(H_expected, S3)
        H_expected = H.vector().array()
    else:
        # use default/default as reference then.
        demag = fk.FKDemag()
        demag.precomputed_bem(bem, boundary_to_global)
        demag.setup(S3, m, Ms, unit_length)
        H_expected = demag.compute_field()
        del(demag)

    if name == "":
        pass
    else:
        name = name + "_"

    runner = create_measurement_runner(S3, m, Ms, unit_length,
                                       H_expected, tol, repetitions,
                                       bem, boundary_to_global)

    solvers = [s[0] for s in df.krylov_solver_methods()]
    preconditioners = [p[0] for p in df.krylov_solver_preconditioners()]

    results_1, failed_1 = runner("first linear solve",
                                 "phi_1_solver", solvers,
                                 "phi_1_preconditioner", preconditioners,
                                 skip,
                                 "{}timings_log_1.txt".format(name),
                                 "{}results_1.pickled".format(name))

    results_2, failed_2 = runner("second linear solve",
                                 "phi_2_solver", solvers,
                                 "phi_2_preconditioner", preconditioners,
                                 skip,
                                 "{}timings_log_2.txt".format(name),
                                 "{}results_2.pickled".format(name))

    return solvers, preconditioners, results_1, failed_1, results_2, failed_2


def fastest_run(results, solvers):
    return _extremum(results, solvers, min)


def fastest_runs(results, solvers):
    return _extremum(results, solvers, min, per_solver=True)


def slowest_run(results, solvers):
    return _extremum(results, solvers, max)


def slowest_runs(results, solvers):
    return _extremum(results, solvers, max, per_solver=True)


def _extremum(results, solvers, func, per_solver=False):
    fastest = []
    for solver in solvers:
        p = func(results[solver].keys(), key=lambda x: results[solver][x])
        fastest.append((solver, p, results[solver][p]))
    if per_solver:
        return fastest
    return func(fastest, key=lambda x: x[2])


def _runs_with_preconditioner(results, solvers, preconditioner):
    indices = []
    times = []
    for i, solver in enumerate(solvers):
        try:
            times.append(results[solver][preconditioner])
        except KeyError:
            pass  # Run with this combination didn't succeed.
        else:
            indices.append(i)  # Will allow us to skip bars in the column chart.
    return times, np.array(indices)


def column_chart(results, solvers, offset=None):
    times_slowest = [r[2] for r in slowest_runs(results, solvers)]
    times_fastest, names_fastest = zip(* [(r[2], r[1]) for r in fastest_runs(results, solvers)])
    times_without_prec, iwop = _runs_with_preconditioner(results, solvers, "none")
    times_with_default_prec, idp = _runs_with_preconditioner(results, solvers, "default")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.2
    ind = np.arange(len(results))

    ax.axhline(results['default']['default'], color=(0.3, 0.3, 0.3), ls=":", zorder=0)
    ax.bar(width + idp, times_with_default_prec, width, color=(0.3, 0.3, 0.3), label="default prec.")
    ax.bar(2 * width + iwop, times_without_prec, width, color=(0.8, 0.8, 0.8), label="no prec.")
    ax.bar(3 * width + ind, times_slowest, width, color="red", label="slowest prec.")
    rects_fastest = ax.bar(ind, times_fastest, width, color="green", label="fastest prec.")
    for i, rect in enumerate(rects_fastest):
        height = rect.get_height()
        offset = offset if offset is not None else 1.05 * height
        ax.text(rect.get_x() + rect.get_width() / 2.0, height + offset, names_fastest[i], ha='center', va='bottom', rotation=90)

    ax.set_xlabel("Solver")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.set_xticks(ind + 2 * width)
    xtickNames = plt.setp(ax, xticklabels=solvers)
    plt.setp(xtickNames, rotation=45)
    return fig
