import os
import numpy as np
import dolfin as df
from finmag.energies import Demag
from finmag.util.convert_mesh import convert_mesh
from finmag.util.helpers import stats, sphinx_sci as s
from finmag.util.magpar import compare_field_directly, compute_demag_magpar

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

table_delim   = "    " + "=" * 10 + (" " + "=" * 30) * 4 + "\n"
table_entries = "    {:<10} {:<30} {:<30} {:<30} {:<30}\n"

def setup_finmag():
    mesh = df.Mesh(convert_mesh(MODULE_DIR + "sphere.geo"))

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m = df.Function(S3)
    m.assign(df.Constant((1, 0, 0)))
    Ms = 1

    demag = Demag()
    demag.setup(S3, m, Ms, unit_length=1e-9)
    H = demag.compute_field()

    return dict(m=m, H=H, Ms=Ms, S3=S3, table=start_table())

def teardown_finmag(finmag):
    finmag["table"] += table_delim
    with open(MODULE_DIR + "table.rst", "w") as f:
        f.write(finmag["table"])

def start_table():
    table  = ".. _demag_table:\n\n"
    table += ".. table:: Summary of comparison of the demag field\n\n"
    table += table_delim 
    table += table_entries.format(
        ":math:`\,`", # hack because sphinx light table syntax does not allow an empty header
        ":math:`\\subn{\\Delta}{test}`",
        ":math:`\\subn{\\Delta}{max}`",
        ":math:`\\bar{\\Delta}`",
        ":math:`\\sigma`")
    table += table_delim
    return table

def pytest_funcarg__finmag(request):
    finmag = request.cached_setup(setup=setup_finmag, teardown=teardown_finmag,
            scope="module")
    return finmag

def test_using_analytical_solution(finmag):
    """ Expecting (-1/3, 0, 0) as a result. """
    REL_TOLERANCE = 2e-2

    H = finmag["H"].reshape((3, -1))
    H_ref = np.zeros(H.shape) 
    H_ref[0] -= 1.0/3.0

    diff = np.abs(H - H_ref)
    rel_diff = diff / np.sqrt(np.max(H_ref[0]**2 + H_ref[1]**2 + H_ref[2]**2))

    finmag["table"] += table_entries.format(
        "analytical", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))

    print "comparison with analytical results, H, relative_difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

def test_using_nmag(finmag):
    REL_TOLERANCE = 5e-5

    H = finmag["H"].reshape((3, -1))
    H_nmag = np.array(zip(* np.genfromtxt(MODULE_DIR + "H_demag_nmag.txt")))
    diff = np.abs(H - H_nmag)
    rel_diff = diff / np.sqrt(np.max(H_nmag[0]**2 + H_nmag[1]**2 + H_nmag[2]**2))

    finmag["table"] += table_entries.format(
        "nmag", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))
    print "comparison with nmag, H, relative_difference:"
    print stats(rel_diff)

    # Compare nmag with analytical solution
    H_ref = np.zeros(H_nmag.shape) 
    H_ref[0] -= 1.0/3.0

    nmag_diff = np.abs(H_nmag - H_ref)
    nmag_rel_diff = nmag_diff / np.sqrt(np.max(H_ref[0]**2 + H_ref[1]**2 + H_ref[2]**2))
    finmag["table"] += table_entries.format(
        "nmag/an.", "", s(np.max(nmag_rel_diff)), s(np.mean(nmag_rel_diff)), s(np.std(nmag_rel_diff)))
    print "comparison beetween nmag and analytical solution, H, relative_difference:"
    print stats(nmag_rel_diff)

    # rel_diff beetween finmag and nmag
    assert np.max(rel_diff) < REL_TOLERANCE

def test_using_magpar(finmag):
    REL_TOLERANCE = 10.0

    magpar_nodes, magpar_H = compute_demag_magpar(finmag["m"], Ms=finmag["Ms"])
    _, _, diff, rel_diff = compare_field_directly(
            finmag["S3"].mesh().coordinates(), finmag["H"],
            magpar_nodes, magpar_H)

    finmag["table"] += table_entries.format(
        "magpar", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))
    print "comparison with magpar, H, relative_difference:"
    print stats(rel_diff)

    # Compare magpar with analytical solution
    H_magpar = magpar_H.reshape((3, -1))
    H_ref = np.zeros(H_magpar.shape) 
    H_ref[0] -= 1.0/3.0

    magpar_diff = np.abs(H_magpar - H_ref)
    magpar_rel_diff = magpar_diff / np.sqrt(np.max(H_ref[0]**2 + H_ref[1]**2 + H_ref[2]**2))

    finmag["table"] += table_entries.format(
        "magpar/an.", "", s(np.max(magpar_rel_diff)), s(np.mean(magpar_rel_diff)), s(np.std(magpar_rel_diff)))
    print "comparison beetween magpar and analytical solution, H, relative_difference:"
    print stats(magpar_rel_diff)

    # rel_diff beetween finmag and magpar
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    f = setup_finmag()
    Hx, Hy, Hz = f["H"].reshape((3, -1))
    print "Expecting (Hx, Hy, Hz) = (-1/3, 0, 0)."
    print "demag field x-component:\n", stats(Hx)
    print "demag field y-component:\n", stats(Hy)
    print "demag field z-component:\n", stats(Hz)

##    test_using_analytical_solution(f)
##    test_using_nmag(f)
    test_using_magpar(f)

    teardown_finmag(f)
