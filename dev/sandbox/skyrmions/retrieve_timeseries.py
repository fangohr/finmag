from dolfin import *

mesh = Box(0,0,0,30e-9,30e-9,3e-9,10,10,1)
V = VectorFunctionSpace(mesh, "CG", 1)
m = Function(V)

series = TimeSeries("solution/m")
times = series.vector_times()

output = File("vtk-files/skyrmion.pvd")
for t in times:
    vector = Vector()
    series.retrieve(vector, t)
    m.vector()[:] = vector

    output << m
