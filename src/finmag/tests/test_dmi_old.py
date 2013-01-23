import numpy as np
import dolfin as df
from finmag.energies import DMI_Old

nm=1e-9
simplexes = 10
length=20*nm
mesh = df.BoxMesh(0,0,0,length,3*nm, 3*nm, simplexes, 1, 1)
V = df.VectorFunctionSpace(mesh, "Lagrange", 1)

def test_DMI_Old_field():
    """
    Simulation 1 is computing H_DMI_Old=dE_dM via assemble.
    Simulation 2 is computing H_DMI_Old=g*M with a suitable pre-computed matrix g.
    Simulation 3 is computing g using a petsc matrix.

    We show that the three methods give equivalent results (this relies
    on H_DMI_Old being linear in M).

    """
    m_initial = df.Expression((
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0'), L=length)
    m = df.interpolate(m_initial, V)
    dmi1 = DMI_Old(D=5e-3, method="box-assemble")
    dmi1.setup(V, m, 8.6e5)
    dmi2 = DMI_Old(D=5e-3, method="box-matrix-numpy")
    dmi2.setup(V, m, 8.6e5)
    dmi3 = DMI_Old(D=5e-3, method="box-matrix-petsc")
    dmi3.setup(V, m, 8.6e5)

    H_dmi1 = dmi1.compute_field()
    H_dmi2 = dmi2.compute_field()
    H_dmi3 = dmi3.compute_field()

    diff12 = np.max(np.abs(H_dmi1 - H_dmi2))
    diff13 = np.max(np.abs(H_dmi1 - H_dmi3))

    print "Difference between H_dmi1 and H_dmi2: max(abs(H_dmi1-H_dmi2))=%g" % diff12
    print "Max value = %g, relative error = %g " % (max(H_dmi1), diff12/max(H_dmi1))
    print "Difference between H_dmi1 and H_dmi3: max(abs(H_dmi1-H_dmi3))=%g" % diff13
    print "Max value = %g, relative error = %g " % (max(H_dmi1), diff13/max(H_dmi1))

    assert diff12 < 1e-8
    assert diff13 < 1e-8
    assert diff12/max(H_dmi1)<1e-15
    assert diff13/max(H_dmi1)<1e-15

if __name__=="__main__":
    test_dmi_field()
