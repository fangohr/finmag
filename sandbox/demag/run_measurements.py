import pickle
import numpy as np
import finmag.energies.demag.fk_demag as fk
import matplotlib.pyplot as plt
from finmag.util.timings import default_timer


def create_measurement_runner(S3, m, Ms, unit_length, H_expected=None, tol=1e-3, bem=None, boundary_to_global=None):
    def runner(timed_method_name, s_param, solvers, p_param, preconditioners, full_timings_log=None, results_cache=None):
        results, failed = [], []

        for timer in (default_timer, fk.fk_timer):
            timer.reset()

        if results_cache is not None:
            try:
                with open(results_cache, "r") as f:
                    results = pickle.load(f)
                    print "Results loaded from file."
            except IOError:
                print "No recorded results found. Will run benchmarks."
                print "Time shown for {} in s.\n".format(timed_method_name)

                if full_timings_log is not None:
                    log = open(full_timings_log, "w")

                # Print the table header.
                columns = [15]
                print " " * columns[0] + "|",
                for prec in preconditioners:
                    width = max(10, len(prec))
                    columns.append(width)
                    print "{:>{w}}".format(prec, w=width),
                print "\n" + "-" * columns[0] + "|" + "-" * (len(columns) + sum(columns[1:]))

                for solver in solvers:
                    # Start a table row.
                    print "{:>{w}}|".format(solver, w=columns[0]),

                    # Compute the demagnetising field with the current solver and each of the preconditioners.
                    results_for_this_solver = []
                    for i, prec in enumerate(preconditioners):
                        demag = fk.FKDemag()
                        if bem is not None:
                            # Use the pre-computed BEM if it was provided.
                            demag.precomputed_bem(bem, boundary_to_global)
                        demag.parameters[s_param] = solver
                        demag.parameters[p_param] = prec
                        demag.setup(S3, m, Ms, unit_length)
                        try:
                            for j in xrange(10):  # Repeat to average out little fluctuations.
                                H = demag.compute_field()  # This can fail with some method/preconditioner combinations.
                        except RuntimeError as e:
                            default_timer.stop_last()
                            failed.append({'solver': solver, 'preconditioner': prec, 'message': e.message})
                            print "{:>{w}}".format("x", w=columns[i + 1]),
                        else:
                            max_diff = np.max(np.abs(H - H_expected))
                            if max_diff > tol:
                                print "{:>{w}}".format("x", w=columns[i + 1]),
                                failed.append({'solver': solver, 'preconditioner': prec, 'message': "error too high", 'error': max_diff})
                            else:
                                measured_time = fk.fk_timer.time_per_call(timed_method_name, "FKDemag")
                                print "{:>{w}.3g}".format(measured_time, w=columns[i + 1]),
                                results_for_this_solver.append({'solver': solver, 'preconditioner': prec, 'time': measured_time})
                        log.write("\nTimings for {} with {}.\n".format(solver, prec))
                        log.write(fk.fk_timer.report() + "\n")
                        fk.fk_timer.reset()
                        del(demag)

                    # Finish the table row with a newline.
                    print ""

                    results.append(results_for_this_solver)

                if results_cache is not None:
                    with open(results_cache, "w") as f:
                        pickle.dump(results, f)

                if full_timings_log is not None:
                    log.close()

        default = results[0][0]['time']
        fastest = fastest_run(results)

        print "\nDefault combination ran in {:.3} s.".format(default)
        print "Fastest combination {}/{} ran in {:.3} s.".format(fastest['solver'], fastest['preconditioner'], fastest['time'])
        print "That is an {:.1%} improvement.".format(1 - fastest['time'] / default)

        return results, failed
    return runner


def fastest_run(results):
    return _extremum(results, min)


def fastest_runs(results):
    return _extremum(results, min, per_solver=True)


def slowest_run(results):
    return _extremum(results, max)


def slowest_runs(results):
    return _extremum(results, max, per_solver=True)


def _extremum(results, func, per_solver=False):
    fastest_per_solver = [func(results_per_solver, key=lambda r: r['time']) for results_per_solver in results]
    if per_solver:
        return fastest_per_solver
    return func(fastest_per_solver, key=lambda r: r['time'])


def runs_with_default_preconditioner(results):
    r = [results_per_solver[0] for results_per_solver in results]
    assert r[0]['preconditioner'] == 'default'
    return r


def runs_without_preconditioner(results):
    r = [results_per_solver[1] for results_per_solver in results]
    assert r[0]['preconditioner'] == "none"
    return r


def column_chart(results):
    times_slowest = [r['time'] for r in slowest_runs(results)]
    times_without_prec = [r['time'] for r in runs_without_preconditioner(results)]
    times_with_default_prec = [r['time'] for r in runs_with_default_preconditioner(results)]

    fastest = fastest_runs(results)
    times_fastest, precs_fastest = zip(* [(r['time'], r['preconditioner']) for r in fastest])

    fig = plt.figure()
    ax = fig.add_subplot(111)

    width = 0.2
    ind = np.arange(len(results))

    rects_fastest = ax.bar(ind, times_fastest, width, color="green", label="fastest prec.")
    ax.bar(width + ind, times_with_default_prec, width, color=(0.3, 0.3, 0.3), label="default prec.")
    ax.bar(2 * width + ind, times_without_prec, width, color=(0.8, 0.8, 0.8), label="no prec.")
    rects_slowest = ax.bar(3 * width + ind, times_slowest, width, color="red", label="slowest prec.")
    ax.axhline(times_with_default_prec[0], color=(0.3, 0.3, 0.3), ls=":", zorder=0)

    ax.set_xlabel("Solver")
    ax.set_ylabel("Time (s)")
    ax.legend()
    ax.set_xticks(ind + 2 * width)
    solvers = [r[0]['solver'] for r in results]
    xtickNames = plt.setp(ax, xticklabels=solvers)
    plt.setp(xtickNames, rotation=45)

    for i, rect in enumerate(rects_fastest):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, 1.05 * height, precs_fastest[i], ha='center', va='bottom', rotation=90)

    return fig
