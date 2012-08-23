"""
This module contains convenience functions to create common types of
meshes. The execution time may be relatively slow (in particular for
fine meshes) because the mesh creation is done externally via Netgen.

It might be nice to reimplement some of these using Dolfin-internal
mesh functions so that they are faster.

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
import tempfile
from dolfin import Mesh

logger = logging.getLogger(name='finmag')

def convert_mesh(geofile, xmlfile=None):
    """
    Convert a .geo file to a .xml.gz file compatible with Dolfin.
    The resulting file is placed in the same directory as inputfile,
    unless specified.

    *Arguments*
        geofile (str)
            Filename of a .geo file which is compatible with netgen.
        xmlfile (str) [optional]
            Filename of generated .xml.gz file which is compatible with Dolfin.
            If no name is given, the generated mesh file will have the same
            basename as the original .geo file.

    *Return*
        xmlfile
            Complete filename of generated mesh. See Example.

    *Example*
        This example shows the simple case where one converts the file
        "myfile.geo" to a dolfin compatible file "myfile.xml.gz".

        .. code-block:: python

            from finmag.util.meshes import convert_mesh
            convert_mesh("myfile.geo")

        Another example shows that this function could be given directly
        as input to Dolfin.Mesh. In this case, the resulting mesh
        is stored in the same directory under the name "mymesh.xml.gz".

        .. code-block:: python

            mesh = Mesh(convert_mesh("myfile.geo", "mymesh.xml.gz"))
            plot(mesh, interactive=True)

    .. Note::

        If an xmlfile happens to exist, it is returned unless the corresponding
        geofile is newer than the xmlfile.

    """
    if xmlfile is None:
        xmlfile = os.path.splitext(geofile)[0] + ".xml.gz"
    else:
        if ".xml.gz" not in xmlfile:
            xmlfile += ".xml.gz"

    if os.path.isfile(xmlfile) and os.path.getctime(xmlfile) > os.path.getctime(geofile):
        logger.debug("The mesh %s already exists, and is automatically returned." % xmlfile)
        return xmlfile

    diffpackfile = run_netgen(geofile)
    xmlfile = convert_diffpack_to_xml(diffpackfile)
    gzipped_xmlfile = compress(xmlfile)

    return gzipped_xmlfile

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
    meshfile = convert_mesh(geofile)
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
    tmpmeshfile = convert_mesh(f.name)
    return Mesh(tmpmeshfile)

# TODO: This function duplicates functionality of the function
#       'spherical_mesh' in convert_mesh.py. It would be nice to unify
#       them. The main difference is that the latter saves the mesh to
#       a file, whereas this one returns the mesh directly. In
#       general, it might be helpful to have two flavours for each
#       mesh-creating function: one which creates a mesh directly and
#       one which writes it to a file (where the former would probably
#       call the latter, only with a temporary file as is done in the
#       function _mesh_from_csg_string above).
def spherical_mesh(radius, maxh):
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
