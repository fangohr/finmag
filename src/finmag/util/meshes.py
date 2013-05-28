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
import copy
import shutil
import commands
import logging
import textwrap
import hashlib
import tempfile
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from types import ListType, TupleType

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
        if os.path.getmtime(result_filename) < os.path.getmtime(geofile):
            # TODO: If save_result is False but the .xml.gz file
            # already exists, it will be updated (hence, saved)
            # anyway. Is this desired?
            logger.warn("The mesh file '{}' is outdated (since it is "
                        "older than the .geo file '{}') and will be "
                        "overwritten.".format(result_filename, geofile))
        else:
            logger.debug("The mesh '{}' already exists and is "
                         "automatically returned.".format(result_filename))
            skip_mesh_creation = True

    if not skip_mesh_creation:
        xml=convert_diffpack_to_xml(run_netgen(geofile))
        change_xml_marker_starts_with_zero(xml)
        result_filename = compress(xml)

    mesh = df.Mesh(result_filename)
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

    A word of caution: It will not overwrite an existing geofile with the same
    name, so remember this when calling the function with save_result=True and
    a custom filename.

    Caveat: if `filename` contains an absolute path then value of
    `directory` is ignored.
    """
    if filename == "":
        filename = hashlib.md5(csg).hexdigest()
    if os.path.isabs(filename) and directory != "":
        logger.warning("Ignoring 'directory' argument (value given: '{}') because 'filename' contains an absolute path: '{}'".format(directory, filename))

    if save_result:
        if directory == "":
            # TODO: Is there a reason why we can't use os.curdir
            # directly as the default in the function definition
            # above? I seem to remember that there was an issue
            # related to the test suite (where files need to be
            # created in MODULE_DIR), but it would be good to
            # double-check that.
            directory = os.curdir

        geofile = os.path.abspath(os.path.join(directory, filename) + ".geo")

        # Make sure that 'directory' actually contains all the
        # directory components of the path:
        directory, _ = os.path.split(geofile)

        if not os.path.exists(directory):
            logger.debug("Creating directory '{}' as it does not exist.".format(directory))
            os.mkdir(directory)


        if not os.path.exists(geofile):
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
    logger.debug("[DDD] NETGENDIR: {}".format(os.environ.get('NETGENDIR')))
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
        logger.warning("Warning: Ignoring netgen's output status of 34304.")
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
             "%s_bi.xml" % basename,
             diffpackfile]
    for f in files:
        if os.path.isfile(f):
            os.remove(f)

    return xmlfile

def change_xml_marker_starts_with_zero(xmlfile):
    """
    the xml file also contains mesh_value_collection in dolfin 1.1 (not in dolfin 1.0) and
    the marker index starts with 1 but the default df.dx refers to dx(0), so this function is
    going to fix this problem (could we report this as a very small bug? seems that dolfin
    community will abandon netegn later?)
    """

    f=open(xmlfile,'r')
    data=f.read()
    f.close()

    data_begin=False
    values=[]

    for line in data.splitlines():

        if 'mesh_value_collection' in line:
            if 'dim="3"' in line:
                data_begin=True
            else:
                data_begin=False

        if data_begin and 'value="' in line:
            v=line.split('value="')[1]
            v=v.split('"')[0]
            values.append(int(v))

    if len(values)==0:
        return

    if min(values)==0:
        return
    elif min(values)<0:
        raise ValueError("Mesh markers are wrong?!")

    min_index=min(values)

    f=open(xmlfile,'w')
    data_begin=False
    for line in data.splitlines():

        if 'mesh_value_collection' in line:
            if 'dim="3"' in line:
                data_begin=True
            else:
                data_begin=False

        if data_begin and 'value="' in line:
            v=line.split('value="')
            v_bak=v[0]
            v=v[1].split('"')[0]
            v=int(v)-min_index
            f.write(v_bak+ 'value="%d"/>\n'%v)
        else:
            f.write(line+'\n')

    f.close()

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
    also the 'native' Dolfin method dolfin.cpp.BoxMesh() which creates a
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
        filename = "box-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(x0, x1, x2, y0, y1, y2, maxh).replace(".", "_")
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
        filename = "sphere-{:g}-{:g}".format(r, maxh).replace(".", "_")
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
        filename = "cyl-{:.1f}-{:.1f}-{:.1f}".format(r, h, maxh).replace(".", "_")
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
        filename = "ellcyl-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(r1, r2, h, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)

def ellipsoid(r1, r2, r3, maxh, save_result=True, filename='', directory=''):
    """
    Return a dolfin mesh representing an ellipsoid with main axes lengths
    r1, r2, r3. The argument `maxh` controls the maximal element size in
    the mesh (see the Netgen manual 4.x, Chapter 2).

    If `save_result` is True (the default), both the generated geofile and
    the dolfin mesh will be saved to disk. By default, the filename will be
    automatically generated based on the values of `r1`, `r2, `h` and `maxh`
    (for example, 'ellipsoid-50_0-25_0-10_0-0_2.geo'), but a different one can be
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
        filename = "ellipsoid-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(r1, r2, r3, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)

def ring(r1,r2, h, maxh, save_result=True, filename='', directory='',with_middle_plane=False):
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
              and plane (0, 0, {h}; 0, 0, 1);

	solid ring = fincyl2 and not fincyl -maxh = {maxh};
        tlo ring;
        """).format(r1=r1,r2=r2, h=h/2.0, maxh=maxh)

    if with_middle_plane:
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
        filename = "ring-{:.1f}-{:.1f}-{:.1f}-{:.1f}".format(r1,r2, h, maxh).replace(".", "_")
    return from_csg(csg_string, save_result=save_result, filename=filename, directory=directory)

def mesh_volume(mesh):
    """
    Computes the total volume of all tetrahedral cells in the mesh.
    alternatively,  volume = assemble(Constant(1)*dx, mesh=mesh)
    """
    return sum([c.volume() for c in df.cells(mesh)])

def nodal_volume(space, unit_length=1):
    """
    Computes the volume of each node of the mesh of the provided (Vector)FunctionSpace.

    The returned numpy.array will be compatible to functions of that FunctionSpace,
    so will be one-dimensional for a FunctionSpace, and three-dimensional
    for a VectorFunctionSpace.

    """
    v = df.TestFunction(space)
    dim = space.mesh().topology().dim()
    if isinstance(space, df.VectorFunctionSpace):
        return df.assemble(df.dot(v, df.Constant((1, 1, 1))) * df.dx).array() * unit_length ** dim
    return df.assemble(v * df.dx).array() * unit_length ** dim

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


def plot_mesh(mesh, scalar_field=None, ax=None, figsize=None, dg_fun=None,**kwargs):
    """
    Plot the given mesh.

    Note that for fine meshes it may be necessary to adjust the
    `linewidth` argument because if the mesh edges are drawn too thick
    compared to the entire mesh then the figure will appear all black.

    FIXME: For 2D meshes we currently draw a wireframe mesh by default
    because I haven't figured out yet how to use `tripcolor` properly.
    This should be changed so that 2D and 3D plotting are consistent.
    Also, once this is done it might be nice to provide a `wireframe`
    keyword which enables/disables wireframe-style plotting.

    TODO: It might be nice to automatically adjust the linewidth, e.g.
          based on the ratio mesh.num_cells()/mesh_volume(mesh).

    *Arguments*

    scalar_field: None or array (of scalar vertex values) or function

        If given, the triangle colors will be derived from the field
        values (using the specified cmap). In this case the `color`
        argument is ignored. If scalar_field is a function, it is
        first applied to all vertices and should expect an array of
        (x, y) (for 2D meshes) or (x, y, z) (for 3d meshes)
        coordinates as its argument.

    ax : None or matplotlib.axes.AxesSubplot (for 2D meshes)
              or matplotlib.axes.Axes3DSubplot (for 3D meshes)

        If `ax` is not given, an appropriate Axes object is created
        automatically. Note that a 3D Axes object can be created as
        follows:

           import matplotlib.pyplot as plt
           ax = plt.gca(projection='3d')

    figsize : pair of floats

        Size of the figure in which the mesh is to be plotted. If the
        `ax` argument is provided, this is ignored.

    All other keyword arguments are passed on to matplotlib's `plot_trisurf`
    (for 3D meshes) or to `triplot` (for 2D meshes). The following defaults
    are used:

       color = 'blue'
       cmap = matplotlib.cm.jet

    *Returns*

    The Axes object in which the mesh was plotted (either the one
    provided by the user or the one which was automatically created).
    """
    dim = mesh.topology().dim()

    # If the user doesn't explicitly specify a linewidth, we
    # heuristically adapt it so that the plot doesn't appear all black
    # because the lines are drawn too thick.
    #
    # There are certainly better ways to do this, but this seems to
    # work reasonably well for most cases. (However, for very oblong
    # structures it may make more sense to check the extent in each
    # dimension individually rather than the mesh volume as a whole.)
    if not kwargs.has_key('linewidth'):
        lw_threshold = 500.0 if dim == 2 else 5000.0
        a = mesh.num_cells()/mesh_volume(mesh)
        if a > lw_threshold:
            kwargs['linewidth'] = pow(lw_threshold / a, 1.0/dim)
            logger.debug("Automatically adapting linewidth to improve plot quality "
                         "(new value: linewidth = {})".format(kwargs['linewidth']))

    # Set default values for some keyword arguments
    if not kwargs.has_key('color'):
        kwargs['color'] = 'blue'
    if kwargs.has_key('cmap'):
        if scalar_field is None:
            kwargs.pop('cmap')
            logger.warning("Ignoring 'cmap' argument since no 'scalar_field' "
                           "argument was given")
    else:
        if scalar_field != None:
            # cmap should only be set when a field was given
            kwargs['cmap'] = cm.jet

    # Create Axis if none was provided
    if ax == None:
        logger.debug("Creating new figure with figsize '{}'".format(figsize))
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(aspect='equal', projection=(None if (dim == 2) else '3d'))
        # XXX                  ^--- Note that equal aspect ratios don't seem
        #     to work for 3d plots yet. See [1], [2].
        # [1] http://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio
        # [2] http://comments.gmane.org/gmane.comp.python.matplotlib.general/27415
    else:
        if figsize != None:
            logger.warning("Ignoring argument `figsize` because `ax` was "
                           "provided explicitly.")

    if dg_fun==None:
        dg_fun=df.Function(df.FunctionSpace(mesh, 'DG', 0))
        dg_fun.vector()[:]=1

    if dim == 2:
        coords = mesh.coordinates()
        x = coords[:,0]
        y = coords[:,1]
        triangs = np.array([[v.index() for v in df.vertices(s)] for s in df.faces(mesh)])

        xmid = x[triangs].mean(axis=1)
        ymid = y[triangs].mean(axis=1)

        zfaces=np.array([dg_fun(xmid[i],ymid[i]) for i in range(len(xmid))])

        if scalar_field != None:
            logger.warning("Ignoring the 'scalar_field' argument as this is not implemented for 2D meshes yet.")

        ## XXX TODO: It would be nice to have the triangles coloured.
        ## This should be possible using 'tripcolor', but I haven't
        ## figured out yet how to pass it the color information (e.g.,
        ## uniformly coloured if we just want to plot the mesh, or
        ## passing an array of color values if we want to plot a
        ## scalar function on a mesh).
        #ax.tripcolor(x, y, triangles=triangs)
        ax.tripcolor(x, y, triangles=triangs, facecolors=zfaces, edgecolors='k', **kwargs)

    elif dim == 3:
        # TODO: Remove this error message once matplotlib 1.3 has been released!
        import matplotlib
        if matplotlib.__version__[:3] < '1.3':
            raise NotImplementedError(
                "Plotting 3D meshes is only supported with versions of "
                "matplotlib >= 1.3.x. Unfortunately, the latest stable "
                "release is 1.2.0, so you have to install the development "
                "version manually. Apologies for the inconvenience!")

        bm = df.BoundaryMesh(mesh, 'exterior')
        coords = bm.coordinates()

        x = coords[:, 0]
        y = coords[:, 1]
        z = coords[:, 2]

        if scalar_field != None:
            try:
                scalar_field = np.array(map(scalar_field, coords))
            except TypeError:
                scalar_field = np.array(scalar_field)

        # Set shade = False by default because it looks nicer
        if not kwargs.has_key('shade'):
            kwargs['shade'] = False

        triangs = [[v.index() for v in df.vertices(s)] for s in df.faces(bm)]
        try:
            ax.plot_trisurf(x, y, z, triangles=triangs, vertex_vals=scalar_field, **kwargs)
        except AttributeError:
            if scalar_field != None:
                logger.debug("Ignoring 'scalar_field' argument because this "
                             "version of matplotlib doesn't support it.")
            ax.plot_trisurf(x, y, z, triangles=triangs, **kwargs)
    else:
        raise ValueError("Plotting is only supported for 2- and 3-dimensional meshes.")

    return ax


def plot_mesh_with_paraview(mesh, **kwargs):
    """
    Use Paraview to render the mesh. This is just a convenience
    function which saves the mesh to a temporary .vtu file and then
    calls `finmag.util.visualization.render_paraview_scene` on that
    file. Any keyword arguments given are passed on to
    `render_paraview_scene` - see its docstring for more information.
    Use the `diffuse_color` keyword to set the mesh color and the
    `outfile` keyword to give a output filename in case the plot should
    be saved to a PNG file.
    """
    import tempfile
    from finmag.util.visualization import render_paraview_scene
    tmpdir = tempfile.mkdtemp()
    tmp_meshfile_pvd = os.path.join(tmpdir, 'mesh.pvd')
    tmp_meshfile_vtu = os.path.join(tmpdir, 'mesh000000.vtu')
    F = df.File(tmp_meshfile_pvd)
    F << mesh
    diffuse_color = kwargs.pop('diffuse_color', [0, 0.9, 0])
    image = render_paraview_scene(
        tmp_meshfile_vtu, field_name=None, add_glyphs=False,
        rescale_colormap_to_data_range=False, show_colorbar=False,
        diffuse_color=diffuse_color, **kwargs)
    shutil.rmtree(tmpdir)
    return image


def plot_mesh_regions(fun_mesh_regions, regions, colors=None, alphas=None,
                      markers='.', marker_sizes=None, zoom_to_mesh_size=True,
                      ax=None, **kwargs):
    """
    Visualise particular regions/subdomains of a mesh by plotting
    markers at the midpoints of all cells belonging to the specified
    region(s). If multiple regions are to be plotted, different
    colours are used to distinguish them.

    *Arguments*

    fun_mesh_regions : dolfin.CellFunction

        A dolfin.MeshFunction defined on the cells of the mesh. It
        should take integer values which indicate for each cell which
        mesh region it belongs to.

    regions : int or list of ints

        The region(s) to plot.

    colors : single color or list of colors

        Colors to be used for the markers of the individual regions.
        If the number of supplied colors is shorter than the number of
        regions, colors from the beginning of the list will be reused.

    alphas : float or list of floats

        Alpha (= transparency) values to be used for the markers of
        the individual regions. If the number of supplied alpha values
        is shorter than the number of regions to be plotted, values
        from the beginning of the list will be reused.

    markers : single marker or list of markers

        Markers to be used for the markers of the individual regions.
        If the number of supplied markers is shorter than the number
        of regions to be plotted, values from the beginning of the
        list will be reused.

    marker_sizes : float or list of float

        Sizes for the markers of the individual regions.
        If the number of supplied markers is shorter than the number
        of regions to be plotted, values from the beginning of the
        list will be reused.

    zoom_to_mesh_size : boolean

        If this is True then the x-, y- and z-axis limits are
        automatically adjusted to the minimum/maximum x-coordinate of
        the mesh so that the visible region covers the extent of the
        mesh. Note that if not all mesh regions are plotted, this
        means that parts of the plot will appear to be empty.

        The reason for this behaviour is that it can be quite
        confusing if a region appears to fill the entire screen (due
        to matplotlib automatically adjusting the axis limits) when it
        is only supposed to cover a small part of the mesh. If this is
        behaviour is undesired, set `zoom_to_mesh_size` to False. If
        necessary, you can also explicitly call 'ax.set_xlim3d()' (and
        similarly for y and z limits) on the Axis object which is
        returned from this function.

    ax : None or matplotlib.axes.Axes3DSubplot

        If `ax` is not given, an appropriate Axes object is created
        automatically.

    **kwargs

        All keyword arguments are passed on to the matplotlib's
        `scatter3d` function.

    *Returns*

    The Axes object in which the mesh was plotted (either the one
    provided by the user or the one which was automatically created).
    """
    def _ensure_is_list(arg):
        res = arg
        if res == None:
            res = []
        elif not isinstance (arg, (ListType, TupleType)):
            res = [res]
        return res

    regions = _ensure_is_list(regions)
    colors = _ensure_is_list(colors)
    alphas = _ensure_is_list(alphas)
    markers = _ensure_is_list(markers)
    sizes = _ensure_is_list(marker_sizes)

    if not isinstance(regions, (ListType, TupleType)):
        raise TypeError("Argument 'region' must be a single integer "
                        "or a list of integers. "
                        "Got: '{}' ({})".format(regions, type(regions)))

    if ax is None:
        ax = plt.gca(projection='3d')

    mesh = fun_mesh_regions.mesh()
    midpoints = [[c.midpoint() for c in df.cells(mesh)
                  if fun_mesh_regions[c.index()] == r] for r in regions]

    pts = [[(pt.x(), pt.y(), pt.z()) for pt in m] for m in midpoints]

    num_regions = len(regions)

    # TODO: More generic would be to create a dictionary here to which
    # we add a color/alpha/... argument iff colors/alphas/... is not
    # None. This allows us to leave the default to matplotlib if no
    # value was explicitly set by the user (instead of creating an
    # artificial default value).

    def _suppy_args(arg_dict, name, lst, i):
        if lst != []:
            val = lst[i % len(lst)]
            if val != None:
                arg_dict[name] = val

    for i in xrange(num_regions):
        arg_dict = copy.copy(kwargs)
        _suppy_args(arg_dict, 'color', colors, i)
        _suppy_args(arg_dict, 'alpha', alphas, i)
        _suppy_args(arg_dict, 'marker', markers, i)
        _suppy_args(arg_dict, 's', sizes, i)
        pts_region = pts[i]
        ax.scatter3D(*zip(*pts_region), **arg_dict)

    if zoom_to_mesh_size:
        logger.debug("Adjusting axis limits in order to zoom to mesh size")
        coords = mesh.coordinates()
        xs = coords[:, 0]
        ys = coords[:, 1]
        zs = coords[:, 2]
        ax.set_xlim3d(min(xs), max(xs))
        ax.set_ylim3d(min(ys), max(ys))
        ax.set_zlim3d(min(zs), max(zs))

    return ax
