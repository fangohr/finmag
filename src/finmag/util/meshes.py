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
import dolfin as df
import numpy as np
from dolfin import Mesh, assemble, Constant, dx

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

def from_csg(csg, save_result=True, filename="", directory=""):
    """
    Using netgen, returns a dolfin mesh object built from the given CSG string.

    If `save_result` is True (which is the default), both the generated
    geofile and the dolfin mesh are saved to disk. By default, the
    filename will use the md5 hash of the csg string, but can be
    specified by passing a name (without suffix) into `filename`.

    The `directory` argument can be used to control where the
    generated files are saved. The `filename` argument may
    contain path components, too (which are simply appended to
    `directory`).

    Caveat: if `filename` contains an absolute path then value of
    `directory` is ignored.
    """
    if filename == "":
        filename = hashlib.md5(csg).hexdigest()
    if os.path.isabs(filename) and directory != "":
        logger.warning("Ignoring 'directory' argument (value given: '{}') because 'filename' contains an absolute path: '{}'".format(directory, filename))

    if save_result:
        geofile = os.path.join(directory, filename) + ".geo"
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
        logger.warning("Netgen's exit status was 0, but an error seems to have occurred anyway (since Netgen's output contains the word 'error').")
        logger.warning("Netgen output:")
        logger.warning("\n====>")
        logger.warning(output)
        logger.warning("<====\n")
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

def box(x0, x1, x2, y0, y1, y2, maxh, save_result=True, filename='', directory=''):
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

    Note that this function uses Netgen to produce the mesh. There is
    also the 'native' Dolfin method dolfin.cpp.Box() which creates a
    regularly-spaced mesh (whereas the mesh produced by Netgen is more
    irregularly spaced). Depending on the context, either can be
    useful.

    """
    # Make sure that each x_i < y_i
    [x0, y0] = sorted([x0, y0])
    [x1, y1] = sorted([x1, y1])
    [x2, y2] = sorted([x2, y2])

    csg = textwrap.dedent("""\
        algebraic3d
        solid main = orthobrick ( {}, {}, {}; {}, {}, {} ) -maxh = {maxh};
        tlo main;
        """).format(x0, x1, x2, y0, y1, y2, maxh=maxh)
    if save_result == True and filename == '':
        filename = "box-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.6e}".format(x0, x1, x2, y0, y1, y2, maxh).replace(".", "_")
    return from_csg(csg, save_result=save_result, filename=filename, directory=directory)

def sphere(r, maxh, save_result=True, filename='', directory=''):
    """
    Returns a dolfin mesh object describing a sphere with radius `r`
    and given mesh coarseness maxh.

    If `save_result` is True (the default), both the generated geofile
    and the dolfin mesh will be saved to disk. By default, the
    filename will be automatically generated based on the values of
    `r` and `maxh` (for example, 'sphere-10_0-0_2.geo'), but a
    different one can be specified by passing a name (without suffix)
    into `filename`. If `save_result` is False, passing a filename has
    no effect.

    The `directory` argument can be used to control where the files
    should be saved in case no filename is given explicitly.

    """
    csg = textwrap.dedent("""\
        algebraic3d
        solid main = sphere ( 0, 0, 0; {r} ) -maxh = {maxh};
        tlo main;
        """).format(r=r, maxh=maxh)

    if save_result == True and filename == '':
        filename = "sphere-{:.1f}-{:.6e}".format(r, maxh).replace(".", "_")
    return from_csg(csg, save_result=save_result, filename=filename, directory=directory)

def cylinder(r, h, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfin mesh representing a cylinder of radius `r` and
    height `h`. The argument `maxh` controls the maximal element size
    in the mesh (see the Netgen manual 4.x, Chapter 2).

    If `save_result` is True (the default), both the generated geofile
    and the dolfin mesh will be saved to disk. By default, the
    filename will be automatically generated based on the values of
    `r`, `h` and `maxh` (for example, 'cyl-50_0-10_0-0_2.geo'), but a
    different one can be specified by passing a name (without suffix)
    into `filename` If `save_result` is False, passing a filename has
    no effect.

    The `directory` argument can be used to control where the files
    should be saved in case no filename is given explicitly.

    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = cylinder (0, 0, 1; 0, 0, -1; {r} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;
        """).format(r=r, h=h, maxh=maxh)
    if save_result == True and filename == '':
        filename = "cyl-{:.1f}-{:.1f}-{:.6e}".format(r, h, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)

def elliptic_cylinder(r1, r2, h, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfin mesh representing an ellipcit cylinder with semi-major
    axis r1, semi-minor axis r2 and height `h`. The argument `maxh` controls
    the maximal element size in the mesh (see the Netgen manual 4.x, Chapter 2).

    If `save_result` is True (the default), both the generated geofile
    and the dolfin mesh will be saved to disk. By default, the
    filename will be automatically generated based on the values of
    `r1`, `r2, `h` and `maxh` (for example, 'cyl-50_0-25_0-10_0-0_2.geo'), but a
    different one can be specified by passing a name (without suffix)
    into `filename` If `save_result` is False, passing a filename has
    no effect.

    The `directory` argument can be used to control where the files
    should be saved in case no filename is given explicitly.

    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = ellipticcylinder (0, 0, 0; {r1}, 0, 0; 0, {r2}, 0 )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;
        """).format(r1=r1, r2=r2, h=h, maxh=maxh)
    if save_result == True and filename == '':
        filename = "ellcyl-{:.1f}-{:.1f}-{:.1f}-{:.6e}".format(r1, r2, h, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)

def ellipsoid(r1, r2, r3, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfin mesh representing an ellipsoid with main axes lengths
    r1, r2, r3. The argument `maxh` controls the maximal element size in
    the mesh (see the Netgen manual 4.x, Chapter 2).

    If `save_result` is True (the default), both the generated geofile and
    the dolfin mesh will be saved to disk. By default, the filename will be
    automatically generated based on the values of `r1`, `r2, `h` and `maxh`
    (for example, 'cyl-50_0-25_0-10_0-0_2.geo'), but a different one can be
    specified by passing a name (without suffix) into `filename` If `save_result`
    is False, passing a filename has no effect.

    The `directory` argument can be used to control where the files should be
    saved in case no filename is given explicitly.

    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid ell = ellipsoid (0, 0, 0; {r1}, 0, 0; 0, {r2}, 0; 0, 0, {r3}) -maxh = {maxh};
        tlo ell;
        """).format(r1=r1, r2=r2, r3=r3, maxh=maxh)
    if save_result == True and filename == '':
        filename = "ellipsoid-{:.1f}-{:.1f}-{:.1f}-{:.6e}".format(r1, r2, r3, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)

def ring(r1,r2, h, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfin mesh representing a ring with inner radius `r1`, outer 
    radius `r2` and height `h`. The argument `maxh` controls the maximal element size
    in the mesh (see the Netgen manual 4.x, Chapter 2).

    If `save_result` is True (the default), both the generated geofile
    and the dolfin mesh will be saved to disk. By default, the
    filename will be automatically generated based on the values of
    `r`, `h` and `maxh` (for example, 'cyl-50_0-10_0-0_2.geo'), but a
    different one can be specified by passing a name (without suffix)
    into `filename` If `save_result` is False, passing a filename has
    no effect.

    The `directory` argument can be used to control where the files
    should be saved in case no filename is given explicitly.

    """
    csg_string = textwrap.dedent("""\
        algebraic3d
        solid fincyl = cylinder (0, 0, -{h}; 0, 0, {h}; {r1} )
              and plane (0, 0, -{h}; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1);
        solid fincyl2 = cylinder (0, 0, -{h}; 0, 0, 0; {r2} )
              and plane (0, 0, -{h}; 0, 0, -1)
              and plane (0, 0, 0; 0, 0, 1);
        solid fincyl3 = cylinder (0, 0, 0; 0, 0, {h}; {r2} )
              and plane (0, 0, 0; 0, 0, -1)
              and plane (0, 0, {h}; 0, 0, 1);   
	solid ring = (fincyl2 or fincyl3) and not fincyl -maxh = {maxh};
        tlo ring;
        """).format(r1=r1,r2=r2, h=h/2.0, maxh=maxh)
    if save_result == True and filename == '':
        filename = "ring-{:.6e}-{:.6e}-{:.6e}-{:.6e}".format(r1,r2, h, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)

def mesh_volume(mesh):
    """
    Computes the sum of the volumes of all tetrahedral cells in the mesh.
    """
    return assemble(Constant(1)*dx, mesh=mesh)

def mesh_info(mesh):
    """
    Return a string containing some basic information about the mesh
    (such as the number of cells/vertices/interior and surface triangles)
    as well as the distribution of edge lengths.
    """
    # Note: It might be useful for this function to return the 'raw' data
    #       (number of cells, vertices, triangles, edge length distribution,
    #       etc.) instead of a string; this could then be used by another
    #       function generate such an info string (or to print the data
    #       directly). However, until such a need arises we leave it as it is.

    # Remark: the number of surface triangles is computed below as follows:
    #
    #     F_s = 4*C - F_i,
    #
    # where we use the abbreviation:
    #
    #    C = number of cells/tetrahedra
    #    F_i = number of interior facets
    #    F_s = number of surface facets
    #
    # Proof: Suppose that each tetrahedron was separated from its neighbours
    #        by a small distance. Then the number of surface facets F_s would
    #        be exactly 4*C (since each tetrahedron has four surface triangles).
    #        To get the number of surface facets in the "true" mesh (without
    #        space between neighbouring cells), all the facets at which two
    #        tetrahedra are "glued together" (i.e., precisely the interior
    #        facets) need to be subtracted from this because otherwise they
    #        would be counted twice.

    edges = [e for e in df.edges(mesh)]
    facets = [f for f in df.facets(mesh)]
    C = mesh.num_cells()
    F = len(facets)
    F_i = 4*C-F
    F_s = F-F_i
    E = len(edges)
    V = mesh.num_vertices()

    lens = [e.length() for e in df.edges(mesh)]
    vals, bins = np.histogram(lens, bins=20)
    vals = np.insert(vals, 0, 0)  # to ensure that 'vals' and 'bins' have the same number of elements
    vals_normalised = 70.0/max(vals)*vals

    info_string = textwrap.dedent("""\
        ===== Mesh info: ==============================
        {:6d} cells (= volume elements)
        {:6d} facets
        {:6d} surface facets
        {:6d} interior facets
        {:6d} edges
        {:6d} vertices

        ===== Distribution of edge lengths: ===========
        """.format(C, F, F_s, F_i, E, V))

    for (b, v) in zip(bins, vals_normalised):
        info_string += "{:.3f} {}\n".format(b, int(round(v))*'*')

    return info_string

def print_mesh_info(mesh):
    print mesh_info(mesh)
