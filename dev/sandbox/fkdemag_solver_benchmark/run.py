import dolfin as df
from finmag.util.meshes import sphere
from benchmark import run_demag_benchmark

print "List of Krylov subspace methods.\n"
for name, description in df.krylov_solver_methods():
    print "{:<20} {}".format(name, description)

print "\nList of preconditioners.\n"
for name, description in df.krylov_solver_preconditioners():
    print "{:<20} {}".format(name, description)

m = len(df.krylov_solver_methods())
n = len(df.krylov_solver_preconditioners())
print "\nThere are {} solvers, {} preconditioners and thus {} diferent combinations of solver and preconditioner.".format(m, n, m * n)

print "\nSolving first system.\n"

ball = sphere(20.0, 1.2, directory="meshes")
m_ball = df.Constant((1, 0, 0))
unit_length = 1e-9
print "The used mesh has {} vertices.".format(ball.num_vertices())
H_expected_ball = df.Constant((-1.0/3.0, 0.0, 0.0))
tol = 0.002
repetitions = 10
solvers, preconditioners, b1, b2 = run_demag_benchmark(m_ball, ball, unit_length, tol, repetitions, "ball", H_expected_ball)

print "\nSolving second system.\n"

film = box(0, 0, 0, 500, 50, 1, maxh=2.0, directory="meshes")
m_film = df.Constant((1, 0, 0))
unit_length = 1e-9
print "The used mesh has {} vertices.".format(film.num_vertices())
tol = 1e-6
repetitions = 10
solvers, preconditioners, f1, f2 = run_measurements(m_film, film, unit_length, tol, repetitions, "film", H_expected_film)



