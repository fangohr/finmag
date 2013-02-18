import dolfin as df
import numpy as np
from finmag.native import sundials
from finmag.util.timings import default_timer
from finmag.energies import Demag

if __name__ == '__main__':
    x0 = y0 = z0 = 0
    x1 = 500e-9
    y1 = 500e-9
    z1 = 2e-9
    nx = 120
    ny = 120
    nz = 3
    mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz)

    print mesh.num_vertices()

    Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)

    Ms = 8.6e5
    expr = df.Expression(('1+x[0]', '1+2*x[1]','1+3*x[2]'))
    m = df.project(expr, Vv)
    m = df.project(df.Constant((1, 0, 0)), Vv)


    demag = Demag("FK")
    demag.setup(Vv, m, Ms)
    demag.compute_field()
    print default_timer
