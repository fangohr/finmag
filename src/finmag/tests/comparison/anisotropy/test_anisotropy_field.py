import os
import dolfin as df
import numpy as np
from finmag.util.convert_mesh import convert_mesh
from finmag.sim.llg import LLG
from finmag.sim.helpers import vectors, stats, sphinx_sci as s

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

Ms = 0.86e6; K1 = 520e3; a = (1, 0, 0);
x1 = y1 = z1 = 20; # same as in bar_5_5_5.geo file

table_delim   = "    " + "=" * 10 + (" " + "=" * 30) * 4 + "\n"
table_entries = "    {:<10} {:<30} {:<30} {:<30} {:<30}\n"

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0) # x, y and z as a fraction
    y = np.maximum(np.minimum(r[1]/y1, 1.0), 0.0) # between 0 and 1 in the 
    z = np.maximum(np.minimum(r[2]/z1, 1.0), 0.0)
    mx = (2 - y) * (2 * x - 1) / 4
    mz = (2 - y) * (2 * z - 1) / 4 
    my = np.sqrt(1 - mx**2 - mz**2)
    return np.array([mx, my, mz])

def start_table():
    table  = ".. _anis_table:\n\n"
    table += ".. table:: Comparison of the anisotropy field computed with finmag against nmag, oommf and magpar\n\n"
    table += table_delim 
    table += table_entries.format(
        ":math:`\,`", # hack because sphinx light table syntax does not allow an empty header
        ":math:`\\subn{\\Delta}{test}`",
        ":math:`\\subn{\\Delta}{max}`",
        ":math:`\\bar{\\Delta}`",
        ":math:`\\sigma`")
    table += table_delim
    return table

def setup_finmag():
    mesh = df.Mesh(convert_mesh(MODULE_DIR + "/bar_5_5_5.geo"))
    llg = LLG(mesh, unit_length=1e-9)
    coords = np.array(zip(* mesh.coordinates()))
    m0 = m_gen(coords).flatten()

    llg.set_m(m0)
    llg.Ms = Ms
    llg.add_uniaxial_anisotropy(K1, df.Constant(a))
    llg.setup(use_exchange=False)

    H_anis = df.Function(llg.V)
    H_anis.vector()[:] = llg._anisotropies[0].compute_field()
    return dict(m=llg.m, H=H_anis, table=start_table(), llg=llg)

def teardown_finmag(finmag):
    finmag["table"] += table_delim
    with open(MODULE_DIR + "table.rst", "w") as f:
        f.write(finmag["table"])

def pytest_funcarg__finmag(request):
    finmag = request.cached_setup(setup=setup_finmag,
            teardown=teardown_finmag, scope="module")
    return finmag

def test_nmag(finmag):
    REL_TOLERANCE = 6e-2

    m_ref = np.genfromtxt(MODULE_DIR + "m0_nmag.txt")
    m_computed = vectors(finmag["m"])
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(MODULE_DIR + "H_anis_nmag.txt")
    H_computed = vectors(finmag["H"].vector().array())
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    print "finmag m x H:"
    print m_cross_H_computed
    print "nmag m x H:"
    print m_cross_H_ref

    mxH = m_cross_H_computed.flatten()
    mxH_ref = m_cross_H_ref.flatten()
    diff = np.abs(mxH - mxH_ref)
    print "comparison with nmag, m x H, difference:"
    print stats(diff)

    rel_diff = diff/ np.sqrt(np.max(mxH_ref[0]**2 + mxH_ref[1]**2 + mxH_ref[2]**2))

    finmag["table"] += table_entries.format(
        "nmag", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))

    print "comparison with nmag, m x H, relative difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

def test_oommf(finmag):
    from finmag.util.oommf import mesh, oommf_uniaxial_anisotropy
    from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf

    REL_TOLERANCE = 2e-5

    oommf_mesh = mesh.Mesh((2, 2, 2), size=(5e-9, 5e-9, 5e-9))
    oommf_anis  = oommf_uniaxial_anisotropy(oommf_m0(lambda r: m_gen(np.array(r)*1e9), oommf_mesh), Ms, K1, a).flat
    finmag_anis = finmag_to_oommf(finmag["H"], oommf_mesh, dims=3)

    assert oommf_anis.shape == finmag_anis.shape
    diff = np.abs(oommf_anis - finmag_anis)
    rel_diff = diff / (np.max(oommf_anis[0]**2 + oommf_anis[1]**2 + oommf_anis[2]**2))

    finmag["table"] += table_entries.format(
        "oommf", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))

    print "comparison with oommf, H, relative_difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

def test_magpar(finmag):
    from finmag.tests.magpar.magpar import compute_anis_magpar, compare_field_directly

    REL_TOLERANCE = 8e-3

    llg = finmag["llg"]
    magpar_nodes, magpar_anis = compute_anis_magpar(llg.V, llg._m, K1, a, Ms, unit_length=1)
    _, _, diff, rel_diff = compare_field_directly(
            llg.mesh.coordinates(), finmag["H"].vector().array(),
            magpar_nodes, magpar_anis)

    finmag["table"] += table_entries.format(
        "magpar", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))
    print "comparison with magpar, H, relative_difference:"

    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    f = setup_finmag()
    test_nmag(f)
    test_oommf(f)
    test_magpar(f)
    teardown_finmag(f)
