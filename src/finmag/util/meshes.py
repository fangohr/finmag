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
import types
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
            Controls whether the resulting dolfin mesh is saved to
            disk (default: True). Doing so greatly speeds up later
            calls to the function with the same geofile. If the
            geofile has been modified since the last run of the mesh
            generation, the saved version is disregarded. The file
            will have the same basename as the geofile, just with the
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
    result_file_exists = False
    skip_mesh_creation = False

    if os.path.isfile(result_filename):
        result_file_exists = True
        if os.path.getctime(result_filename) < os.path.getctime(geofile) and save_result==False:
            # TODO: If save_result is False but the .xml.gz file already exists, it will be updated (hence, saved) anyway. Is this desired?
            logger.warn("The mesh file '{}' is outdated (since it is older than the .geo file '{}') and will be overwritten.".format(result_filename, geofile))
        else:
            logger.debug("The mesh %s already exists, and is automatically returned." % result_filename)
            skip_mesh_creation = True

    if not skip_mesh_creation:
        result_filename = compress(convert_diffpack_to_xml(run_netgen(geofile)))

    mesh = Mesh(result_filename)
    if not save_result and not result_file_exists:
        # We delete the .xml.gz file only if it didn't exist previously
        os.remove(result_filename)
        logger.debug("Removing file '%s' because mesh is created on the fly." % result_filename)
    return mesh

def _normalize_filepath(dirname, filename):
    """
    Construct a "normalized" pair (d,f) from the input arguments, where d is
    the directory component (without the trailing slash) and f is a filename.

    If `dirname` is None, `filename` is allowed to contain path components
    (which can be relative or absolute). If dirname is None (or empty) and
    filename does *not* contain any path components, dirname is set to the
    current directory (= os.curdir). If `directory` is not None, `filename`
    must be a simple string without any path components.
    """
    if not isinstance(dirname, (str, types.NoneType)):
        raise TypeError("'dirname' must be a string or None (got: '%s' of type '%s')".format(dirname, type(dirname)))
    if not isinstance(filename, str):
        raise TypeError("'filename' must be a string (got: '%s' of type '%s')".format(filename, type(filename)))
    if filename == '':
        raise ValueError("'filename' must not be empty")

    fdir = os.path.dirname(filename)
    if dirname != None and fdir != "":
        raise ValueError("'dirname' must be None if 'filename' contains a path component. Values given: dirname='{}', filename='{}'".format(dirname, filename))

    d = os.path.normpath(dirname or fdir or os.curdir)
    f = os.path.basename(filename)
    return (d,f)

def from_csg(csg, save_result=True, filename=None, directory=None):
    """
    Using netgen, returns a dolfin mesh object built from the given CSG string.

    If `save_result` is True (which is the default), both the generated
    geofile and the dolfin mesh are saved to disk. By default, the
    filename will use the md5 hash of the csg string, but can be
    specified by passing a name (without suffix) into `filename`.

    """
    if filename is None:
        filename = hashlib.md5(csg).hexdigest()
    (directory, filename) = _normalize_filepath(directory, filename)

    if save_result:
        geofile = os.path.join(directory, os.path.join(filename + ".geo"))
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
        logger.debug("Removing file '%s' because mesh is created on the fly." % tmp.name)
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
    logger.debug("Compressing {}".format(filename))
    compr_cmd = 'gzip -f %s' % filename
    status, output = commands.getstatusoutput(compr_cmd)
    if status != 0:
        print output
        print "gzip failed with exit code", status
        sys.exit(4)
    return filename + ".gz"

def box(x0, x1, x2, y0, y1, y2, maxh, save_result=True, filename=None, directory=None):
    """
    Returns a dolfin mesh object describing an axis-parallel box.

    The two points (x0, x1, x2) and (y0, y1, y2) are interpreted as
    two diagonally opposite corners of the box.

    If `save_result` is True (the default), both the generated geofile
    and the dolfin mesh will be saved to disk. By default, the
    filename will be automatically generated based on the values of
    `radius` and `maxh` (for example,'box-0_0-0_0-0_0-1_0-2_0-3_0.geo'),
    but a different one can be specified by passing a name (without
    suffix) into `filename`. If `save_result` is False, passing a
    filename has no effect.

    The `directory` argument can be used to control where the files
    should be saved in case no filename is given explicitly.

    """
    # Make sure that each x_i < y_i
    [x0, y0] = sorted([x0, y0])
    [x1, y1] = sorted([x1, y1])
    [x2, y2] = sorted([x2, y2])

    csg = textwrap.dedent("""\
        algebraic3d
        solid main = orthobrick ( {}, {}, {}; {}, {}, {} ) -maxh = {maxh};
        tlo main;""").format(x0, x1, x2, y0, y1, y2, maxh=maxh)
    if save_result == True and filename is None:
        filename = "box-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(x0, x1, x2, y0, y1, y2, maxh).replace(".", "_")
    return from_csg(csg, save_result=save_result, filename=filename, directory=directory)

def sphere(radius, maxh, save_result=True, filename=None, directory=None):
    """
    Returns a dolfin mesh object describing a sphere with given radius and mesh coarseness.

    If `save_result` is True (the default), both the generated geofile
    and the dolfin mesh will be saved to disk. By default, the
    filename will be automatically generated based on the values of
    `radius` and `maxh` (for example, 'sphere-10_0-0_2.geo'), but a
    different one can be specified by passing a name (without suffix)
    into `filename`. If `save_result` is False, passing a filename has
    no effect.

    The `directory` argument can be used to control where the files
    should be saved in case no filename is given explicitly.

    """
    csg = textwrap.dedent("""\
        algebraic3d
        solid main = sphere ( 0, 0, 0; {radius} ) -maxh = {maxh};
        tlo main;""").format(radius=radius, maxh=maxh)

    if save_result == True and filename is None:
        filename = "sphere-{:.1f}-{:.1f}".format(radius, maxh).replace(".", "_")
    return from_csg(csg, save_result=save_result, filename=filename, directory=directory)

def cylinder(radius, height, maxh, save_result=True, filename=None, directory=None):
    """
    Return a dolfin mesh representing a cylinder of radius `radius`
    and height `height`. `maxh` controls the maximal element size in
    the mesh (see the Netgen manual 4.x, Chapter 2).

    If `save_result` is True (the default), both the generated geofile and
    the dolfin mesh will be saved to disk. By default, the filename will be
    automatically generated based on the values of `radius`, `height`
    and `maxh` (for example, 'cyl-50_0-10_0-0_2.geo'), but a different
    one can be specified by passing a name (without suffix) into `filename`
    If `save_result` is False, passing a filename has no effect.

    The `directory` argument can be used to control where the files
    should be saved in case no filename is given explicitly.

    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = cylinder (0, 0, 1; 0, 0, -1; {radius} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {height}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;""").format(radius=radius, height=height, maxh=maxh)
    if save_result == True and filename is None:
        filename = "cyl-{:.1f}-{:.1f}-{:.1f}".format(radius, height, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)
