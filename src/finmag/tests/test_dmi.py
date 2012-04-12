import numpy as np
import dolfin as df
from finmag.sim.llg import LLG
from finmag.sim.helpers import components

nm=1e-9
simplexes = 10
length=20*nm
mesh = df.Box(0,0,0,length,3*nm, 3*nm, simplexes, 1, 1)


def test_dmi_field_box_assemble_equal_box_matrix():
    """Simulation 1 is computing H_dmi=dE_dM via assemble.
    Simulation 2 is computing H_dmi=g*M with a suitable pre-computed matrix g.
    
    Here we show that the two methods give equivalent results (this relies
    on H_dmi being linear in M).
    """
    m_initial = (
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0')
    llg1 = LLG(mesh)
    llg1.set_m(m_initial, L=length)
    llg1.setup(use_dmi=True,dmi_method='box-matrix-numpy')
    llg1.solve()
    H_dmi1 = llg1.H_dmi

    llg2 = LLG(mesh)
    llg2.set_m(m_initial, L=length)
    llg2.setup(use_dmi=True,dmi_method='box-assemble')
    llg2.solve()
    H_dmi2 = llg2.H_dmi

    diff = max(abs(H_dmi1-H_dmi2))
    print "Difference between H_dmi1 and H_dmi2: max(abs(H_dmi1-H_dmi2))=%g" % diff
    print "Max value = %g, relative error = %g " % (max(H_dmi1), diff/max(H_dmi1))
    assert diff < 1e-8
    assert diff/max(H_dmi1)<1e-15



def test_dmi_field_box_matrix_numpy_same_as_box_matrix_petsc():
    """Simulation 1 is computing H_dmi=g*M  via box-matrix-numpy.
    Simulation 2 is computing g using a petsc matrix.

    Apart from memory and speed differences, this should do the same thing.
    """
    m_initial = (
            '(2*x[0]-L)/L',
            'sqrt(1 - ((2*x[0]-L)/L)*((2*x[0]-L)/L))',
            '0')
    llg1 = LLG(mesh)
    llg1.set_m(m_initial, L=length)
    llg1.setup(use_dmi=True,dmi_method='box-matrix-numpy')
    llg1.solve()
    H_dmi1 = llg1.H_dmi

    llg2 = LLG(mesh)
    llg2.set_m(m_initial, L=length)
    llg2.setup(use_dmi=True,dmi_method='box-matrix-petsc')
    llg2.solve()
    H_dmi2 = llg2.H_dmi

    diff = max(abs(H_dmi1-H_dmi2))
    print "Difference between H_dmi1 and H_dmi2: max(abs(H_dmi1-H_dmi2))=%g" % diff
    print "Max value = %g, relative error = %g " % (max(H_dmi1), diff/max(H_dmi1))
    assert diff < 1e-8
    assert diff/max(H_dmi1)<1e-15



if __name__=="__main__":
    test_dmi_field_box_assemble_equal_box_matrix()
    test_dmi_field_box_matrix_numpy_same_as_box_matrix_petsc()
