import os, sys, textwrap, commands, logging

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

            from finmag.util.convert_mesh import convert_mesh
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
