import os
import shutil
import tempfile
import textwrap
from py.test import raises
from finmag.util.meshes import *
from finmag.util.meshes import _normalize_filepath
from dolfin import Mesh, cells, assemble, Constant, dx
from math import pi

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 0.05
BOX_TOLERANCE = 1e-10 # tolerance for the box() method, which should be much more precise

def test_normalize_filepath():
    assert(_normalize_filepath(None, 'foo.txt') == (os.curdir, 'foo.txt'))
    assert(_normalize_filepath('', 'foo.txt') == (os.curdir, 'foo.txt'))
    assert(_normalize_filepath(None, 'bar/baz/foo.txt') == ('bar/baz', 'foo.txt'))
    assert(_normalize_filepath(None, '/bar/baz/foo.txt') == ('/bar/baz', 'foo.txt'))
    assert(_normalize_filepath('/bar/baz', 'foo.txt') == ('/bar/baz', 'foo.txt'))
    assert(_normalize_filepath('/bar/baz/', 'foo.txt') == ('/bar/baz', 'foo.txt'))
    with raises(TypeError):
        _normalize_filepath(42, None)
    with raises(TypeError):
        _normalize_filepath(None, 23)
    with raises(ValueError):
        _normalize_filepath(None, '')
    with raises(ValueError):
        _normalize_filepath('bar', '')
    with raises(ValueError):
        _normalize_filepath('bar', 'baz/foo.txt')

# Note: the test below is disabled for the time being because it adds
# some time to the execution of the test suite but doesn't provide
# much benefit (apart from checking a few corner cases). So it can
# probably be deleted.
def disabled_test_from_geofile_and_from_csg():
    radius = 1.0
    maxh = 0.5

    tmpdir = tempfile.mkdtemp()
    tmpfile = tempfile.NamedTemporaryFile(suffix='.geo', dir=tmpdir, delete=False)

    csg_string = textwrap.dedent("""\
        algebraic3d
        solid main = sphere (0, 0, 0; {radius}) -maxh = {maxh};
        tlo main;""").format(radius=radius, maxh=maxh)

    try:
        # Create a temporay .geo file which contains the geometric
        # description of a sphere.
        tmpfile.write(csg_string)
        tmpfile.close()

        geofile = tmpfile.name
        xmlfile = os.path.splitext(geofile)[0] + ".xml.gz"

        # Create a Dolfin mesh from the .geo file, first without saving the result
        mesh1 = from_geofile(geofile, save_result=False)
        assert(isinstance(mesh1, Mesh))
        assert(not os.path.isfile(xmlfile))

        # Now do the same, but save the result (and check that the .xml.gz file is there)
        mesh2 = from_geofile(geofile, save_result=True)
        assert(isinstance(mesh2, Mesh))
        assert(os.path.isfile(xmlfile))

        # Again, read the mesh form the .geo file, but this time it should be read instantaneously (TODO: how do we check this in the test?)
        mesh3 = from_geofile(geofile, save_result=True)
        assert(isinstance(mesh3, Mesh))
        assert(os.path.isfile(xmlfile))

        # Create a mesh from a CSG string directly
        mesh4 = from_csg(csg_string, save_result=False)

        # Check that the volume of the sphere is approximately correct
        vol_exact = 4.0/3*pi*radius**2
        for mesh in [mesh1, mesh2, mesh3, mesh4]:
            vol_mesh = assemble(Constant(1)*dx, mesh=mesh)
            assert(abs(vol_mesh - vol_exact)/vol_exact < TOLERANCE)
    finally:
        tmpfile.close()
        shutil.rmtree(tmpdir)

def test_box():
    # We deliberately choose the two corners so that x1 > y1, to see
    # whether the box() function can cope with this.
    (x0, x1, x2) = (-0.2, 1.4, 3.0)
    (y0, y1, y2) = (1.1, 0.7, 2.2)

    maxh = 10.0  # large value so that we get as few vertices as possible

    # Note: We use 'save_result=False' in this test so that we can also test 'anonymous'
    #       mesh creation. In the other tests, we use 'save_result=True' so that the mesh
    #       is loaded from a file for faster execution.
    mesh = box(x0, x1, x2, y0, y1, y2, maxh=maxh, save_result=False)
    print "Num nodes: %s" % mesh.num_vertices()

    vol_exact = abs((y0-x0)*(y1-x1)*(y2-x2))
    vol_mesh = assemble(Constant(1)*dx, mesh=mesh)

    assert(abs(vol_mesh - vol_exact)/vol_exact < BOX_TOLERANCE)

def test_sphere():
    radius = 1.0
    maxh = 0.2

    mesh = sphere(radius=radius, maxh=maxh, save_result=True, directory=MODULE_DIR)
    print "Num nodes: %s" % mesh.num_vertices()

    vol_exact = 4.0/3*pi*radius**2
    vol_mesh = assemble(Constant(1)*dx, mesh=mesh)
    assert(abs(vol_mesh - vol_exact)/vol_exact < TOLERANCE)

def test_cylinder():
    radius = 1.0
    height = 2.0
    maxh = 0.2

    mesh = cylinder(radius=radius, height=height, maxh=maxh, save_result=True, directory=MODULE_DIR)
    print "Num nodes: %s" % mesh.num_vertices()

    vol_exact = pi*radius**2*height
    vol_mesh = assemble(Constant(1)*dx, mesh=mesh)
    assert(abs(vol_mesh - vol_exact)/vol_exact < TOLERANCE)
