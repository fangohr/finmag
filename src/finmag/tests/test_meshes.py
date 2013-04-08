import os
import time
import shutil
import tempfile
import textwrap
from finmag.util.meshes import *
from dolfin import Mesh
from math import pi
from StringIO import StringIO

import logging
logger = logging.getLogger("finmag")

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
TOLERANCE = 0.05
BOX_TOLERANCE = 1e-10 # tolerance for the box() method, which should be much more precise

def test_from_geofile_and_from_csg():
    radius = 1.0
    maxh = 0.3

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

        # Capture logging output in a string-stream
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        logger.addHandler(handler)

        # Read the mesh form the .geo file again, but this time it
        # should be read instantaneously.
        mesh3 = from_geofile(geofile, save_result=True)
        assert(isinstance(mesh3, Mesh))
        assert(os.path.isfile(xmlfile))
        handler.flush()
        #assert(stream.getvalue() == "The mesh '{}' already exists and is "
        #       "automatically returned.\n".format(xmlfile))

        # 'Touch' the .geo file so that it is newer than the .xml.gz
        # file. Then check that upon reading the mesh the .xml.gz file
        # is recreated. Note that most machines have sub-millisecond
        # precision in their timestamps, but on some systems (such as
        # osiris) only full seconds seem to be stored. So we wait for
        # one second to make sure that the .geo file is picked up as
        # being newer.
        stream.truncate(0)  # clear stream
        time.sleep(1)
        os.utime(geofile, None)  # update the 'last modified' timestamp
        mesh4 = from_geofile(geofile, save_result=True)
        assert(isinstance(mesh4, Mesh))
        assert(os.path.isfile(xmlfile))
        handler.flush()
        assert(stream.getvalue().startswith("The mesh file '{}' is outdated "
               "(since it is older than the .geo file '{}') and will "
               "be overwritten.\n".format(xmlfile, geofile)))

        # Create a mesh from a CSG string directly
        mesh5 = from_csg(csg_string, save_result=False)

        # Check that the volume of the sphere is approximately correct
        vol_exact = 4.0/3*pi*radius**2
        for mesh in [mesh1, mesh2, mesh3, mesh4, mesh5]:
            vol_mesh = mesh_volume(mesh)
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
    vol_exact = abs((y0-x0)*(y1-x1)*(y2-x2))
    vol_mesh = mesh_volume(mesh)
    assert(abs(vol_mesh - vol_exact)/vol_exact < BOX_TOLERANCE)

def test_sphere():
    r = 1.0
    maxh = 0.2

    mesh = sphere(r=r, maxh=maxh, save_result=True, directory=MODULE_DIR)
    vol_exact = 4.0/3*pi*r**2
    vol_mesh = mesh_volume(mesh)
    assert(abs(vol_mesh - vol_exact)/vol_exact < TOLERANCE)

def test_cylinder():
    r = 1.0
    h = 2.0
    maxh = 0.2

    mesh = cylinder(r=r, h=h, maxh=maxh, save_result=True, directory=MODULE_DIR)
    vol_exact = pi*r*r*h
    vol_mesh = mesh_volume(mesh)
    assert(abs(vol_mesh - vol_exact)/vol_exact < TOLERANCE)

def test_elliptic_cylinder():
    r1 = 2.0
    r2 = 1.0
    h = 2.5
    maxh = 0.2

    mesh = elliptic_cylinder(r1=r1, r2=r2, h=h, maxh=maxh, save_result=True, directory=MODULE_DIR)
    vol_exact = pi*r1*r2*h
    vol_mesh = mesh_volume(mesh)
    assert(abs(vol_mesh - vol_exact)/vol_exact < TOLERANCE)

def test_ellipsoid():
    r1 = 2.0
    r2 = 1.0
    r3 = 0.5
    maxh = 0.2

    mesh = ellipsoid(r1=r1, r2=r2, r3=r3, maxh=maxh, save_result=True, directory=MODULE_DIR)
    vol_exact = 4.0/3*pi*r1*r2*r3
    vol_mesh = mesh_volume(mesh)
    assert(abs(vol_mesh - vol_exact)/vol_exact < TOLERANCE)

def test_plot_mesh_regions():
    """
    This test simply calls the function
    `finmag.util.meshes.plot_mesh_regions` to see if it can be called
    with different arguments without error. No checking of the output
    figure is done whatsoever.
    """

    # Write csg string to a temporary file
    mesh = from_geofile(os.path.join(MODULE_DIR, "sphere_in_cube.geo"))
    mesh_regions = df.MeshFunction("uint", mesh, os.path.join(MODULE_DIR, "sphere_in_cube_mat.xml"))

    # Call plot_mesh_regions with a variety of different arguments
    ax = plot_mesh_regions(mesh_regions, regions=1)
    plot_mesh_regions(mesh_regions, regions=1, colors="green", ax=ax)
    plot_mesh_regions(mesh_regions, regions=[1, 2], zoom_to_mesh_size=False)
    plot_mesh_regions(mesh_regions, regions=[1, 2], colors=["green", "red"],
                      alphas=[1.0, 0.25])
    plot_mesh_regions(mesh_regions, regions=[1, 2], colors=["green", "red"],
                      alphas=[1.0, 0.25], markers=['<', 'H'], marker_sizes=[20, 50])
    # 'None' inside a list means: use matplotlib's default
    plot_mesh_regions(mesh_regions, regions=[1, 2], colors=[None, "red"],
                      alphas=[0.3, None], marker_sizes=[None, 100])
    # Sanity check (empty regions)
    plot_mesh_regions(mesh_regions, regions=[])
