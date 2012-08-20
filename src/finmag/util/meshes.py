"""
This module contains convenience functions to create common types of
meshes. The execution time may be relatively slow (in particular for
fine meshes) because the mesh creation is done externally via Netgen.

It might be nice to reimplement some of these using Dolfin-internal
mesh functions so that they are faster.
"""

import os, re, shutil, tempfile, logging, textwrap
from dolfin import Mesh
from convert_mesh import convert_mesh

logger = logging.getLogger(name='finmag')

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
              and plane (0, 0, {height_min}; 0, 0, -1)
              and plane (0, 0, {height_max}; 0, 0, 1) -maxh = {maxh};
        tlo fincyl;""").format(radius=radius, height_min=-0.5*height, height_max=+0.5*height, maxh=maxh)

    return _mesh_from_csg_string(csg_string)
