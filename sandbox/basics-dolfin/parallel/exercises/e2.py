import dolfin as df
from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()

def fun(x):
    rank = comm.Get_rank()
    return rank*0.1

class HelperExpression(df.Expression):
    def __init__(self,value):
        super(HelperExpression, self).__init__()
        self.fun = value

    def eval(self, value, x):
        value[0] = self.fun(x)



if __name__=="__main__":

    mesh = df.RectangleMesh(0, 0, 20, 20, 20, 20)
    V = df.FunctionSpace(mesh, 'DG', 0)
    hexp = HelperExpression(fun)
    u = df.interpolate(hexp, V)
    
    file = df.File('color_map.pvd')
    file << u