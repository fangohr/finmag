import os
import numpy as np
from dolfin import Mesh
from finmag.sim.llg import LLG
from finmag.util.convert_mesh import convert_mesh
from finmag.sim.helpers import stats
from finmag.tests.magpar.magpar import compare_field_directly, compute_demag_magpar

MODULE_DIR = os.path.dirname(os.path.abspath(__file__)) + "/"

def setup_finmag():
    mesh = Mesh(convert_mesh(MODULE_DIR + "sphere.geo"))
    llg = LLG(mesh, mesh_units=1e-9)
    llg.set_m((1, 0, 0))
    llg.Ms = 1
    llg.setup(use_demag=True)
    return dict(H=llg.demag.compute_field().reshape((3, -1)), llg=llg)

def teardown_finmag(finmag):
    pass

def pytest_funcarg__finmag(request):
    finmag = request.cached_setup(setup=setup_finmag, teardown=teardown_finmag,
            scope="module")
    return finmag

def test_using_analytical_solution(finmag):
    """ Expecting (-1/3, 0, 0) as a result. """
    REL_TOLERANCE = 2e-2

    H_ref = np.zeros(finmag["H"].shape) 
    H_ref[0] -= 1.0/3.0

    diff = np.abs(finmag["H"] - H_ref)
    rel_diff = diff / np.sqrt(np.max(H_ref[0]**2 + H_ref[1]**2 + H_ref[2]**2))

    print "comparison with analytical results, H, relative_difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

def test_using_nmag(finmag):
    REL_TOLERANCE = 3e-5

    H_ref = np.array(zip(* np.genfromtxt(MODULE_DIR + "H_demag_nmag.txt")))
    diff = np.abs(finmag["H"] - H_ref)
    rel_diff = diff / np.sqrt(np.max(H_ref[0]**2 + H_ref[1]**2 + H_ref[2]**2))

    print "comparison with nmag, H, relative_difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

def test_using_magpar(finmag):
    REL_TOLERANCE = 3e-1

    llg = finmag["llg"]
    magpar_nodes, magpar_H = compute_demag_magpar(llg.V, llg._m, llg.Ms)
    _, _, diff, rel_diff = compare_field_directly(
            llg.mesh.coordinates(), finmag["H"].flatten(),
            magpar_nodes, magpar_H)

    print "comparison with magpar, H, relative_difference:"
    print stats(rel_diff)
    assert np.max(rel_diff) < REL_TOLERANCE

if __name__ == "__main__":
    f = setup_finmag()

    Hx, Hy, Hz = f["H"] 
    print "Expecting (Hx, Hy, Hz) = (-1/3, 0, 0)."
    print "demag field x-component:\n", stats(Hx)
    print "demag field y-component:\n", stats(Hy)
    print "demag field z-component:\n", stats(Hz)

    test_using_analytical_solution(f)
    test_using_nmag(f)
    test_using_magpar(f)
