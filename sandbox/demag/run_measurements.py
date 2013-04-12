import pickle
import numpy as np
import finmag.energies.demag.fk_demag as fk
import matplotlib.pyplot as plt
from finmag.util.timings import default_timer


def _table_header(labels):
    # Print the table header.
    columns = [15]
    print " " * columns[0] + "|",
    for label in labels:
        width = max(10, len(label))
        columns.append(width)
        print "{:>{w}}".format(label, w=width),
    print "\n" + "-" * columns[0] + "|" + "-" * (len(columns) + sum(columns[1:]))
    return columns


def _row_start(label, width):
    print "{:<{w}}|".format(label, w=width),


def _row_entry(entry, width):
    print "{:>{w}.3}".format(entry, w=width),


def _prepare_demag_object(S3, m, Ms, unit_length, s_param, solver, p_param, preconditioner, bem, boundary_to_global):
    demag = fk.FKDemag()
    if bem is not None:
        # Use the precomputed BEM if it was provided.
        demag.precomputed_bem(bem, boundary_to_global)
    demag.parameters[s_param] = solver
    demag.parameters[p_param] = preconditioner
    demag.setup(S3, m, Ms, unit_length)
    return demag


def _row_stop():
    print ""


def _loaded_results_table(timed_method_name, results, solvers, preconditioners):
    print "Results loaded from file."
    print "Time shown for {} in s.\n".format(timed_method_name)

    columns = _table_header(preconditioners)
    for i, solver in enumerate(solvers):
        _row_start(solver, columns[0])
        for j, prec in enumerate(preconditioners):
            try:
                time = results[solver][prec]
            except KeyError:
                time = "-"
            _row_entry(time, columns[j + 1])
        _row_stop()

    default = results['default']['default']
    fastest = fastest_run(results, solvers)

    print "\nDefault combination ran in {:.3} s.".format(default)
    print "Fastest combination {}/{} ran in {:.3} s.".format(fastest[0], fastest[1], fastest[2])
    print "That is an {:.1%} improvement.".format(1 - fastest[2] / default)


def create_measurement_runner(S3, m, Ms, unit_length, H_expected=None, tol=1e-3, repeats=10, bem=None, boundary_to_global=None):
    def runner(timed_method_name, s_param, solvers, p_param, preconditioners, skip=[], full_timings_log="timings.txt", results_cache=None):
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

        columns = _table_header(preconditioners)
        for solver in solvers:
            _row_start(solver, columns[0])
            results_for_this_solver = {}
            # Compute the demagnetising field with the current solver and each of the preconditioners.
            for i, prec in enumerate(preconditioners):
                if (solver, prec) in skip:
                    _row_entry("s", columns[i + 1])
                    continue
                demag = _prepare_demag_object(S3, m, Ms, unit_length, s_param, solver, p_param, prec, bem, boundary_to_global)
                try:
                    for j in xrange(repeats):  # Repeat to average out little fluctuations.
                        H = demag.compute_field()  # This can fail with some method/preconditioner combinations.
                except RuntimeError as e:
                    default_timer.stop_last()
                    _row_entry("x", columns[i + 1])
                    failed.append({'solver': solver, 'preconditioner': prec, 'message': e.message})
                else:
                    max_diff = np.max(np.abs(H - H_expected))
                    if max_diff > tol:
                        _row_entry("x", columns[i + 1])
                        failed.append({'solver': solver, 'preconditioner': prec, 'message': "error {:.3} higher than allowed {:.3}"})
                    else:
                        measured_time = fk.fk_timer.time_per_call(timed_method_name, "FKDemag")
                        _row_entry(measured_time, columns[i + 1])
                        results_for_this_solver[prec] = measured_time
                log.write("\nTimings for {} with {}.\n".format(solver, prec))
                log.write(fk.fk_timer.report() + "\n")
                fk.fk_timer.reset()
                del(demag)
            results[solver] = results_for_this_solver
            _row_stop()

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
