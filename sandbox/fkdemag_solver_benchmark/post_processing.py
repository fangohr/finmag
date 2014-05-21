import numpy as np
import dolfin as df
import matplotlib.pyplot as plt

def column_chart(results, solvers, preconditioners, offset=None, ymax=10):
    slowest = results.max(0)
    fastest = results.min(0)
    default = results[0]
    no_prec = results[1]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    width = 0.2
    ind = np.arange(len(solvers))

    ax.axhline(results[0, 0], color=(0.3, 0.3, 0.3), ls=":", zorder=0)
    rects_fastest = ax.bar(ind, fastest, width, color="green", label="fastest prec.")
    ax.bar(width + ind, default, width, color=(0.3, 0.3, 0.3), label="default prec.")
    ax.bar(2 * width + ind, no_prec, width, color=(0.8, 0.8, 0.8), label="no prec.")
    ax.bar(3 * width + ind, slowest, width, color="red", label="slowest prec.")

    # annotate fastest runs with name of preconditioner
    fastest_ind = results.argmin(0)
    for i, rect in enumerate(rects_fastest):
        height = rect.get_height()
        offset = offset if offset is not None else 1.05 * height
        ax.text(rect.get_x() + rect.get_width() / 2.0, height + offset,
                preconditioners[fastest_ind[i]],
                ha='center', va='bottom', rotation=90)

    ax.set_xlabel("method")
    ax.set_ylabel("time (ms)")
    ax.set_ylim((0, ymax))
    ax.legend()
    ax.set_xticks(ind + 2 * width)
    xtickNames = plt.setp(ax, xticklabels=solvers)
    plt.setp(xtickNames, rotation=0)
    return fig

if __name__ == "__main__":
    ms = 1e3
    solvers = [s[0] for s in df.krylov_solver_methods()]
    preconditioners = [p[0] for p in df.krylov_solver_preconditioners()]

    ymax = [[6, 6], [6, 10]]
    for i, system in enumerate(["ball", "film"]):
        for j, potential in enumerate(["1", "2"]):
            results = ms * np.ma.load(system + "_" + potential + ".pickle")

            with open(system + "_" + potential + ".txt", "w") as f:
                f.write("& {} \\\\\n".format(" & ".join(solvers)))
                f.write("\\hline\n")
                for pi, p in enumerate(preconditioners):
                    numbers = ["{:.3}".format(r) for r in results[pi]]
                    f.write("{} & {} \\\\\n".format(p, " & ".join(numbers)))

            fig = column_chart(results, solvers, preconditioners, offset=0.2, ymax=ymax[i][j])
            plt.savefig(system + "_" + potential + ".png")
            plt.close()
