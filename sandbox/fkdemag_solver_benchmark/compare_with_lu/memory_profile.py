import time
import numpy as np
import dolfin as df
from finmag.energies import Demag
from finmag.util.meshes import sphere

@profile
def run_benchmark(solver):
    radius = 10.0
    maxh = 0.5
    unit_length = 1e-9
    mesh = sphere(r=radius, maxh=maxh, directory="meshes")
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    Ms = 1
    m_0 = (1, 0, 0)
    m = df.Function(S3)
    m.assign(df.Constant(m_0))
    H_ref = np.array((- Ms / 3.0, 0, 0))

    if solver == "krylov":
        parameters = {}
        parameters["phi_1_solver"] = "default"
        parameters["phi_1_preconditioner"] = "default"
        parameters["phi_2_solver"] = "default"
        parameters["phi_2_preconditioner"] = "default"
        demag = Demag("FK", solver_type="Krylov", parameters=parameters)
    elif solver == "lu":
        demag = Demag("FK", solver_type="LU")
    else:
        import sys
        print "Unknown solver {}.".format(solver)
        sys.exit(1)

    demag.setup(S3, m, Ms, unit_length)

    REPETITIONS = 10
    start = time.time()
    for j in xrange(REPETITIONS):
        H = demag.compute_field()
    elapsed = (time.time() - start) / REPETITIONS

    H = H.reshape((3, -1))
    spread = abs(H[0].max() - H[0].min()) 
    error = abs(H.mean(axis=1)[0] - H_ref[0]) / abs(H_ref[0])

    print elapsed, error, spread

if __name__ == "__main__":
    run_benchmark("krylov")
