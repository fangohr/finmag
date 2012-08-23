"""
This module contains convenience functions to create common types of
meshes. The execution time may be relatively slow (in particular for
fine meshes) because the mesh creation is done externally via Netgen.

Caveat: Netgen only saves the first 5-6 digits (or so) of each
coordinate during the mesh creation process. Thus it is not advisable
to use these functions to create meshes on the nanoscale. Instead,
create a "macroscopic" mesh and use something like unit_length=1e-9 to
set the desired length scale when reading the mesh into Finmag!
"""

import os
import sys
import commands
import logging
import textwrap
import hashlib
import tempfile
from dolfin import Mesh

logger = logging.getLogger(name='finmag')

def from_geofile(geofile, save_result=True, result_filename=None):
    """
    Using netgen, returns a dolfin mesh object built from the given geofile.

    *Arguments*
        geofile (str)
            Filename of a .geo file which is compatible with netgen.
        save_result (bool) [optional]
            Controls whether the resulting dolfin mesh is saved to disk. Doing
            so greatly speeds up later calls to the function with the 
            same geofile. If the geofile has been modified since the last run
            of the mesh generation, the saved version is disregarded.
        result_filename (str) [optional]
            Filename of the dolfin mesh when it is saved to disk.
            If no name is given, the generated mesh file will have the same
            basename as the original geofile.

    *Return*
        mesh
            dolfin mesh object, instance of the dolfin.cpp.Mesh class.

    *Example*
        The return value of this function can be used like a conventional
        dolfin mesh.

        .. code-block:: python

            import dolfin 
            from finmag.util.meshes import from_geofile
            mesh = from_geofile(path_to_my_geofile)
            dolfin.plot(mesh, interactive=True)

    """
    if result_filename is None:
        result_filename = os.path.splitext(geofile)[0] + ".xml.gz"
    else:
        if ".xml.gz" not in result_filename:
            result_filename += ".xml.gz"

    if os.path.isfile(result_filename) and os.path.getctime(result_filename) > os.path.getctime(geofile):
        logger.debug("The mesh %s already exists, and is automatically returned." % result_filename)
    else:
        result_filename= compress(convert_diffpack_to_xml(run_netgen(geofile)))

    mesh = Mesh(result_filename)
    if not save_result:
        os.remove(result_filename)
    return mesh

def from_csg(csg, save_result=True, result_filename=None):
    """
    Using netgen, returns a dolfin mesh object built from the given CSG string.

    Refer to the documentation for from_geofile. By default, the generated mesh
    is saved to disk, with a filename which is the md5 hash of the csg string.

    """
    tmp = tempfile.NamedTemporaryFile(suffix='.geo', delete=False)
    tmp.write(csg)
    tmp.close()

    if result_filename == None:
        result_filename = hashlib.md5(csg).hexdigest()

    mesh = from_geofile(tmp.name, result_filename=result_filename)

    # Since we used delete=False in NamedTemporaryFile, we are
    # responsible for the deletion of the file.
    os.remove(tmp.name)
    return mesh

def run_netgen(geofile):
    """
    Runs netgen on the geofile and returns a file in DIFFPACK format.

    """
    if not os.path.isfile(geofile):
        raise ValueError("Can't find file {}.".format(geofile))

    basename, extension = os.path.splitext(geofile)
    diffpackfile = basename + ".grid"

    if not extension == ".geo":
        raise ValueError("Input needs to be a .geo file.")

    logger.debug("Using netgen to convert {} to DIFFPACK format.".format(geofile))
    netgen_cmd = "netgen -geofile={} -meshfiletype='DIFFPACK Format' -meshfile={} -batchmode".format(
            geofile, diffpackfile)

    status, output = commands.getstatusoutput(netgen_cmd)
    if status == 34304:
        print "Warning: Netgen output status was 34304, but this seems to be a spurious error that only occurred on Anders Johansen's machine. Ignoring for now..."
    elif status != 0:
        print output
        print "netgen failed with exit code", status
        sys.exit(2)
    elif output.lower().find("error") != -1:
        print "Netgen's exit status was 0, but an error seems to have occurred anyway."
        print "Error message:"
        print "\n====>"
        print output
        print "<====\n"
        sys.exit(2)
    logger.debug('Done!')
    return diffpackfile

def convert_diffpack_to_xml(diffpackfile):
    """
    Converts the diffpackfile to xml using dolfin-convert.

    """
    if not os.path.isfile(diffpackfile):
        raise ValueError("Can't find file {}.".format(diffpackfile))
    logger.debug('Using dolfin-convert to convert {} to xml format.'.format(diffpackfile))

    basename = os.path.splitext(diffpackfile)[0]
    xmlfile = basename + ".xml"
    dolfin_conv_cmd = 'dolfin-convert {0} {1}'.format(diffpackfile, xmlfile)
    status, output = commands.getstatusoutput(dolfin_conv_cmd)
    if status != 0:
        print output
        print "dolfin-convert failed with exit code", status
        sys.exit(3)
 
    files = ["%s.xml.bak" % basename,
             "%s_mat.xml" % basename,
             "%s_bi.xml" % basename,
             diffpackfile]
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

    return xmlfile

def compress(filename):
    """
    Compress file using gzip.

    """
    logger.debug("Compressing {}.".format(filename))
    compr_cmd = 'gzip -f %s' % filename
    status, output = commands.getstatusoutput(compr_cmd)
    if status != 0:
        print output
        print "gzip failed with exit code", status
        sys.exit(4)
    return filename + ".gz"

def spherical_mesh(radius, maxh, directory=""):
    """
    Returns the name of a dolfin compatible meshfile describing
    a sphere with radius radius and maximal mesh size maxh.

    This function is not well behaved by default - it will place
    the meshfile in whatever the current working directory is. Pass
    a directory along to control where the files end up.

    """
    filename = "sphere-{:.1f}-{:.1f}".format(radius, maxh).replace(".", "_")

    meshfile = os.path.join(directory, filename + ".xml.gz")
    if os.path.isfile(meshfile):
        return meshfile

    geofile = os.path.join(directory, filename + ".geo")
    with open(geofile, "w") as f:
        f.write(csg_for_sphere(radius, maxh))
    meshfile = from_geofile(geofile)
    os.remove(geofile)

    return meshfile

def csg_for_sphere(radius, maxh):
    """
    For a sphere with the maximal mesh size maxh (compare netgen manual 4.X page 10)
    and the radius radius, this function will return a string describing the
    sphere in the constructive solid geometry format.

    """
    csg = textwrap.dedent("""\
        algebraic3d
        solid main = sphere ( 0, 0, 0; {radius} ) -maxh = {maxh};
        tlo main;""").format(radius=radius, maxh=maxh)
    return csg

def _mesh_from_csg_string(csg_string):
    """
    This function writes the 'csg_string' (which should contain a
    geometrical description of the mesh in the constructive solid
    geometry format as understood by Netgen) into a .geo file in a
    temporary directory and converts this into a dolfin-readable
    .xml.gz file, which is imported into Dolfin. The resulting Mesh is
    returned.

    This function should only be used internally.
    """
    tmpdir = tempfile.mkdtemp()
    f = tempfile.NamedTemporaryFile(suffix='.geo', delete=False, dir=tmpdir)
    f.write(csg_string)
    f.close()
    mesh = from_geofile(f.name)
    return mesh

# TODO: This function duplicates functionality of the function
#       'spherical_mesh' in from_geofile.py. It would be nice to unify
#       them. The main difference is that the latter saves the mesh to
#       a file, whereas this one returns the mesh directly. In
#       general, it might be helpful to have two flavours for each
#       mesh-creating function: one which creates a mesh directly and
#       one which writes it to a file (where the former would probably
#       call the latter, only with a temporary file as is done in the
#       function _mesh_from_csg_string above).
def sphere(radius, maxh):
    """
    Return a dolfin mesh representing a sphere of radius `radius`.
    `maxh` controls the maximal element size in the mesh (see the
    Netgen manual 4.x, Chapter 2).
    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid main = sphere ( 0, 0, 0; {radius} ) -maxh = {maxh};
        tlo main;""").format(radius=radius, maxh=maxh)

    return _mesh_from_csg_string(csg_string)

def cylindrical_mesh(radius, height, maxh):
    """
    Return a dolfin mesh representing a cylinder of radius `radius`
    and height `height`. `maxh` controls the maximal element size in
    the mesh (see the Netgen manual 4.x, Chapter 2).
    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = cylinder (0, 0, 1; 0, 0, -1; {radius} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {height}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;""").format(radius=radius, height=height, maxh=maxh)

    return _mesh_from_csg_string(csg_string)
