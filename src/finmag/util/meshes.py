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

def from_geofile(geofile, save_result=True):
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
            The file will have the same basename as the geofile, just with the
            extension .xml.gz instead of .geo, and will be placed in 
            the same directory.

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
    result_filename = os.path.splitext(geofile)[0] + ".xml.gz"
    if os.path.isfile(result_filename) and os.path.getctime(result_filename) > os.path.getctime(geofile):
        logger.debug("The mesh %s already exists, and is automatically returned." % result_filename)
    else:
        result_filename = compress(convert_diffpack_to_xml(run_netgen(geofile)))

    mesh = Mesh(result_filename)
    if not save_result:
        os.remove(result_filename)
    return mesh

def from_csg(csg, save_result=True, directory="", name=""):
    """
    Using netgen, returns a dolfin mesh object built from the given CSG string.

    If save_result is True, both the generated geofile and the dolfin mesh
    are saved to disk. By default, the filenames will use the md5 hash
    of the csg string, but can be specified by passing a
    name (without suffix).

    """
    if save_result:
        if name == "":
            name = hashlib.md5(csg).hexdigest()
        geofile = os.path.join(directory, name + ".geo")
        with open(geofile, "w") as f:
            f.write(csg)
        mesh = from_geofile(geofile, save_result=True)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix='.geo', delete=False)
        tmp.write(csg)
        tmp.close()
        mesh = from_geofile(tmp.name, save_result=False)
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

def sphere(radius, maxh, directory=""):
    """
    Returns a dolfin mesh object describing a sphere with given radius and mesh coarseness.

    It will save both the generated geofile and the dolfin mesh to disk.
    Pass a directory into directory, if you want to control where the
    saved files end up.

    """
    csg = textwrap.dedent("""\
        algebraic3d
        solid main = sphere ( 0, 0, 0; {radius} ) -maxh = {maxh};
        tlo main;""").format(radius=radius, maxh=maxh)

    name = "sphere-{:.1f}-{:.1f}".format(radius, maxh).replace(".", "_")
    return from_csg(csg, directory=directory, name=name)

def cylinder(radius, height, maxh, directory=""):
    """
    Return a dolfin mesh representing a cylinder of radius `radius`
    and height `height`. `maxh` controls the maximal element size in
    the mesh (see the Netgen manual 4.x, Chapter 2).

    It will save both the generated geofile and the dolfin mesh to disk.

    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = cylinder (0, 0, 1; 0, 0, -1; {radius} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {height}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;""").format(radius=radius, height=height, maxh=maxh)
    name = "cyl-{:.1f}-{:.1f}-{:.1f}".format(radius, height, maxh).replace(".", "_")
    return from_csg(csg_string, directory=directory, name=name)
