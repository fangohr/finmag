"""
Visuale distribution of degrees of freedom.

Invoke with

    `mpirun -np N python map_dof_rank.py`

to run with N processes. Creates a `color_map.pvd` file which you can open with
paraview. When the --plot option is set, it will also display plots to screen.

"""
import argparse
import dolfin as df
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class RankExpression(df.Expression):
    """
    The value of this expression is equal to the
    MPI rank of the process it is evaluated on.

    """
    def eval(self, value, x):
        value[0] = rank


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=("Visualise distribution of the degrees"
        " of freedom. Will create `color_map.pvd` which you can open with paraview."
        " Can also plot to screen."))
    parser.add_argument('-p', '--plot', action='store_true', help='if set, plots rank to screen')
    args = parser.parse_args()

    if rank == 0:
        print "You are running this example with {} processors.".format(size)

    if size == 1:
        print "To use N processes: `mpirun -np N python {}`.".format(__file__)

    mesh = df.RectangleMesh(0, 0, 20, 20, 20, 20)
    V = df.FunctionSpace(mesh, 'DG', 0)
    rank_function = df.interpolate(RankExpression(), V)
    rank_function.rename("rank", "unique id of process")

    file = df.File('color_map.pvd')
    file << rank_function

    if args.plot:
        title = "rank {} of {}".format(rank, size - 1)
        df.plot(rank_function, interactive=True, title=title)
    

