import os
import numpy as np
import dolfin as df
from finmag.energies import Exchange

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
n = 20
Ms = 8.6e5
A = 1
REL_TOL = 1e-4

def exchange(mesh, unit_length):
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m_expr = df.Expression(("x[1]*u", "0", "sqrt(1-pow(x[1]*u, 2))"), u=unit_length)
    m = df.interpolate(m_expr, S3)
    exch = Exchange(A)
    exch.setup(S3, m, Ms, unit_length=unit_length)
    H = exch.compute_field()
    E = exch.compute_energy()
    return m.vector().array(), H, E

def test_compare_exchange_for_two_dolfin_meshes():
    """
    Check that a mesh expressed in nanometers gives the same results
    as a mesh expressed in meters for the exchange interaction.

    """
    mesh_nm = df.BoxMesh(0, 0, 0, 1, 1, 1, n, n, n) # in nm
    m_nm, H_nm, E_nm = exchange(mesh_nm, unit_length=1e-9)

    mesh = df.BoxMesh(0, 0, 0, 1e-9, 1e-9, 1e-9, n, n, n)
    m, H, E = exchange(mesh, unit_length=1)

    rel_diff_m = np.max(np.abs(m_nm - m)) # norm m = 1
    print "Difference of magnetisation is {:.2f}%.".format(100 * rel_diff_m)
    assert rel_diff_m < REL_TOL

    rel_diff_E = abs((E_nm - E) / E)
    print "Relative difference between E_nm = {:.5g} and E_m = {:.5g} is d = {:.2f}%.".format(E_nm, E, 100*rel_diff_E)
    assert rel_diff_E < REL_TOL

    max_diff_H = np.max(np.abs(H_nm - H)/np.max(H))
    print "Maximum of relative difference between the two fields is d = {:.2f}%.".format(100*max_diff_H)
    assert np.max(np.abs(H)) > 0
    assert max_diff_H < REL_TOL
