import os
import shutil
import tempfile
import textwrap
from py.test import raises
from finmag.util.meshes import *
from finmag.util.meshes import _normalize_filepath
from dolfin import Mesh, cells, assemble, Constant, dx
from math import pi

TOLERANCE = 0.1

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

def test_from_geofile_and_from_csg():
    radius = 1.0
    maxh = 0.2

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
            assert(abs(vol_mesh - vol_exact) < TOLERANCE)
    finally:
        tmpfile.close()
        shutil.rmtree(tmpdir)
