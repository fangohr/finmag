import numpy as np
from dolfin import Mesh, plot
from finmag.util.convert_mesh import convert_mesh
from finmag.demag.solver_gcr import FemBemGCRSolver
from finmag.demag.problems.prob_base import FemBemDeMagProblem

mesh = Mesh(convert_mesh("mesh/sphere10.geo"))
M = ("1.0", "0.0", "0.0")
problem = FemBemDeMagProblem(mesh, M)
solver = FemBemGCRSolver(problem)
phi = solver.solve()
H_demag = solver.get_demagfield(phi)

p = plot(phi, interactive=False)
p.write_png("examples/demag/phi.png")

solver.save_function(phi, "GCR_phi")
solver.save_function(H_demag, "GCR_demagfield")
