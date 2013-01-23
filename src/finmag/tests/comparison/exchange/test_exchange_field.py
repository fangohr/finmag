import os
import dolfin as df
import numpy as np
from finmag.energies import Exchange
from finmag.util.helpers import vectors, norm, stats, sphinx_sci as s

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

x0 = 0; x1 = 20e-9; xn = 10;
Ms = 0.86e6; A = 1.3e-11

table_delim   = "    " + "=" * 10 + (" " + "=" * 30) * 4 + "\n"
table_entries = "    {:<10} {:<30} {:<30} {:<30} {:<30}\n"

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0)
    mx = (2 * x - 1) * 2/3 
    mz = np.sin(2 * np.pi * x) / 2
    my = np.sqrt(1.0 - mx**2 - mz**2)
    return np.array([mx, my, mz])

def start_table():
    table  = ".. _exchange_table:\n\n"
    table += ".. table:: Comparison of the exchange field computed with finmag against nmag and oommf\n\n"
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
    mesh = df.IntervalMesh(xn, x0, x1)
    coords = np.array(zip(* mesh.coordinates()))
 
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    m = df.Function(S3)
    m.vector()[:] = m_gen(coords).flatten()

    exchange = Exchange(A)  
    exchange.setup(S3, m, Ms)

    H_exc = df.Function(S3)
    H_exc.vector()[:] = exchange.compute_field()
    return dict(m=m, H=H_exc, table=start_table())

def teardown_finmag(finmag):
    finmag["table"] += table_delim
    with open(os.path.join(MODULE_DIR, "table.rst"), "w") as f:
        f.write(finmag["table"])

def pytest_funcarg__finmag(request):
    finmag = request.cached_setup(setup=setup_finmag,
            teardown=teardown_finmag, scope="module")
    return finmag

def test_against_nmag(finmag):
    REL_TOLERANCE = 2e-14

    m_ref = np.genfromtxt(os.path.join(MODULE_DIR, "m0_nmag.txt"))
    m_computed = vectors(finmag["m"].vector().array())
    assert m_ref.shape == m_computed.shape

    H_ref = np.genfromtxt(os.path.join(MODULE_DIR, "H_exc_nmag.txt"))
    H_computed = vectors(finmag["H"].vector().array())
    assert H_ref.shape == H_computed.shape

    assert m_ref.shape == H_ref.shape
    m_cross_H_ref = np.cross(m_ref, H_ref)
    m_cross_H_computed = np.cross(m_computed, H_computed)

    diff = np.abs(m_cross_H_ref - m_cross_H_computed)
    rel_diff = diff/max([norm(v) for v in m_cross_H_ref])
 
    finmag["table"] += table_entries.format(
        "nmag", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))

    print "comparison with nmag, m x H, relative difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

def test_against_oommf(finmag):
    REL_TOLERANCE = 8e-2

    from finmag.util.oommf import mesh, oommf_uniform_exchange
    from finmag.util.oommf.comparison import oommf_m0, finmag_to_oommf

    oommf_mesh = mesh.Mesh((xn, 1, 1), size=(x1, 1e-12, 1e-12))
    oommf_exc  = oommf_uniform_exchange(oommf_m0(m_gen, oommf_mesh), Ms, A).flat
    finmag_exc = finmag_to_oommf(finmag["H"], oommf_mesh, dims=1)

    assert oommf_exc.shape == finmag_exc.shape
    diff = np.abs(oommf_exc - finmag_exc)
    rel_diff = diff / np.sqrt(np.max(oommf_exc[0]**2 + oommf_exc[1]**2 + oommf_exc[2]**2))

    finmag["table"] += table_entries.format(
        "oommf", s(REL_TOLERANCE, 0), s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))

    print "comparison with oommf, H, relative_difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == '__main__':
    f = setup_finmag()
    test_against_nmag(f)
    test_against_oommf(f)
    teardown_finmag(f)
