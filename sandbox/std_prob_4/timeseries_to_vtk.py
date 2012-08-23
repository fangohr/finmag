import os
import dolfin as df
from finmag.util.meshes import convert_mesh

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

mesh_file = os.path.join(MODULE_DIR, "bar.geo")
mesh = df.Mesh(convert_mesh(mesh_file))

S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
m = df.Function(S3)

series = df.TimeSeries("solution/m")
times = series.vector_times()

vtk_files = df.File("vtk-files/m.pvd")
for t in times:
    vector = df.Vector()
    series.retrieve(vector, t)
    m.vector()[:] = vector
    
    vtk_files << m
