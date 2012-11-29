import os
import numpy as np
import dolfin as df
from finmag.energies import Exchange
from finmag.util.meshes import box

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def exchange(mesh, cube_length, unit_length):
    A = 13.0e-12
    exch = Exchange(A)

    S3 = df.VectorFunctionSpace(mesh, "CG", 1)
    m = df.project(df.Expression(("0", "sin(x[0]*pi/(2*L))", "cos(x[0]*pi/(2*L))"), L=cube_length), S3)
    Ms = 8.6e5
    exch.setup(S3, m, Ms, unit_length=unit_length)

    H = df.Function(S3)
    H.vector().set_local(exch.compute_field())
    E = exch.compute_energy()

    return H, E

def _test_compare_with_dolfin_mesh():
    """
    Check that a netgen mesh expressed in nanometers gives the same results
    as a dolfin mesh expressed in meters for the exchange interaction.

    """

    L_nm = 10 # nm
    mesh_netgen = box(0, 0, 0, L_nm, L_nm, L_nm, L_nm/5, filename="netgen_box.geo", directory=MODULE_DIR)
    H_netgen, E_netgen = exchange(mesh_netgen, L_nm, unit_length=1e-9)

    L_m = 10e-9
    mesh = df.Box(0, 0, 0, L_m, L_m, L_m, 5, 5, 5)
    H, E = exchange(mesh, L_m, unit_length=1)

    print "Comparing E = {} with netgen and E = {} with dolfin.".format(E_netgen, E)
    assert abs(E_netgen - E) < 1e-19

    rs = np.linspace(0, L_nm, 5)
    for r_nm in rs:
        # probe along cube diagonal
        r_m = r_nm * 1e-9
        print H_netgen(r_nm, r_nm, r_nm)
        print H(r_m, r_m, r_m)
        assert abs(H_netgen(r_nm, r_nm, r_nm) - H(r_m, r_m, r_m)) < 1e-14
    
