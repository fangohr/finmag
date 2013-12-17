import os
import numpy as np
import dolfin as df
from finmag.util.meshes import from_geofile
from finmag.energies import UniaxialAnisotropy, CubicAnisotropy
from finmag.util.helpers import sphinx_sci as s

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def pytest_funcarg__finmag(request):
    finmag = request.cached_setup(setup=setup,
            teardown=teardown, scope="session")
    return finmag

Ms = 0.86e6; K1 = 520e3; K2=230e3; K3=123e3;
u1 = (1, 0, 0);  u2 = (0,1,0)
x1 = y1 = z1 = 20; # same as in bar.geo file

def m_gen(r):
    x = np.maximum(np.minimum(r[0]/x1, 1.0), 0.0) # x, y and z as a fraction
    y = np.maximum(np.minimum(r[1]/y1, 1.0), 0.0) # between 0 and 1 in the 
    z = np.maximum(np.minimum(r[2]/z1, 1.0), 0.0)
    mx = (2 - y) * (2 * x - 1) / 4
    mz = (2 - y) * (2 * z - 1) / 4 
    my = np.sqrt(1 - mx**2 - mz**2)
    return np.array([mx, my, mz])

def setup(K2=K2):
    print "Running finmag..."
    mesh = from_geofile(os.path.join(MODULE_DIR, "bar.geo"))
    coords = np.array(zip(* mesh.coordinates()))

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    m = df.Function(S3)
    m.vector()[:] = m_gen(coords).flatten()
    
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    Ms_cg = df.Function(S1)
    Ms_cg.vector()[:] = Ms

    anisotropy = UniaxialAnisotropy(K1, u1, K2=K2) 
    anisotropy.setup(S3, m, Ms_cg, unit_length=1e-9)

    H_anis = df.Function(S3)
    H_anis.vector()[:] = anisotropy.compute_field()
    return dict(m=m, H=H_anis, S3=S3, table=start_table())

def setup_cubic():
    print "Running finmag..."
    mesh = from_geofile(os.path.join(MODULE_DIR, "bar.geo"))
    coords = np.array(zip(* mesh.coordinates()))

    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1, dim=3)
    m = df.Function(S3)
    m.vector()[:] = m_gen(coords).flatten()
    
    S1 = df.FunctionSpace(mesh, "Lagrange", 1)
    Ms_cg = df.Function(S1)
    Ms_cg.vector()[:] = Ms

    anisotropy = CubicAnisotropy(u1=u1,u2=u2,K1=K1,K2=K2, K3=K3) 
    anisotropy.setup(S3, m, Ms_cg, unit_length=1e-9)

    H_anis = df.Function(S3)
    H_anis.vector()[:] = anisotropy.compute_field()
    return dict(m=m, H=H_anis, S3=S3, table=start_table())

def teardown(finmag):
    write_table(finmag)

table_delim   = "    " + "=" * 10 + (" " + "=" * 30) * 4 + "\n"
table_entries = "    {:<10} {:<30} {:<30} {:<30} {:<30}\n"

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

def write_table(finmag):
    finmag["table"] += table_delim
    with open(os.path.join(MODULE_DIR, "table.rst"), "w") as f:
        f.write(finmag["table"])

def table_entry(name, tol, rel_diff):
    return table_entries.format(
            name, s(tol, 0),
            s(np.max(rel_diff)), s(np.mean(rel_diff)), s(np.std(rel_diff)))
