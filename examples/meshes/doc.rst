Using meshes
============

In this example we demonstrate how to create and import meshes in Finmag
using various different methods.

Dolfin stores meshes in its own XML format. Such a file (which may be
compressed with gzip to disk save space) can be imported directly into
a Dolfin ``Mesh``:

.. code-block:: python

    from dolfin import Mesh
    mesh = Mesh("my_mesh_file.xml.gz")


Alternatively, Finmag provides convenience functions which can
generate Dolfin-compatible meshes via `Netgen <http://www.hpfem.jku.at/netgen/>`_.
There are several high-level ones which create simple shapes like a box, a sphere,
or a cylinder:

.. code-block:: python

    from finmag.util.meshes import *

    mesh1 = box(0.0, 0.0, 0.0, 1.0, 2.0, 3.0, maxh=0.2, save_result=False)
    mesh2 = sphere(2.0, maxh=0.4, save_result=True, directory='mesh_files')
    mesh3 = cylinder(radius=1.5, height=0.5, maxh=0.1, save_result=True, filename='cylinder')

The box in ``mesh1`` is described by the coordinates of two opposite corners, the sphere
in ``mesh2`` is described by its radius and the cylinder in ``mesh3`` is described by its radius and height.
All these methods also require a ``maxh`` parameter which
gives a bound for the maximal size of the elements in the mesh. In addition, all
these methods accept an optional ``save_result`` parameter. If this is True (the
default) then the ``.geo`` specification (in Netgen format) and the Dolfin mesh file
with suffix ``.xml.gz`` are saved to disk (they can be reloaded later, which greatly speeds up the mesh creation process).
The filename and directory where they are saved can be controlled
using the corresponding optional arguments (where ``filename`` should be given without suffix).
If ``filename`` is omitted then a default filename is used based on the parameters
specifying the geometry (for example, in the code snippet above the files for
``mesh2`` would be saved under the names
``mesh_files/sphere-2_0-0_4.geo`` and ``mesh_files/sphere-2_0-0_4.xml.gz``).

Another way to load a mesh into Finmag is to import a ``.geo`` file
containing a description of the mesh geometry in the CSG
(constructive solid geometry) fomat as understood by Netgen. An
example for a ``.geo`` file describing a sphere of radius 1 might look
like this (the ``maxh`` parameter controls the mesh coarseness):

.. literalinclude:: ../src/finmag/tests/sphere-1_0-0_2.geo

The corresponding mesh looks like this:

.. image:: ../examples/meshes/sphere-1_0-0_2.png
    :scale: 75
    :align: center

Assuming that the name of the ``.geo`` file is ``sphere-1_0-0_2.geo``, it
can be imported into Finmag as follows:

.. code-block:: python

    from finmag.utils.meshes import from_geofile
    mesh = from_geofile("sphere-1_0-0_2.geo", save_result=True)

If the optional parameter ``save_result`` is ``True`` (the default)
then the converted Dolfin mesh is saved to a ``.xml.gz`` file (with
the same basename as the ``.geo`` file). If a ``.xml.gz`` file with
the same basename already exists (and it is newer than the
corresponding ``.geo`` file) when ``from_geofile`` is called then the
mesh is simply loaded from the ``.xml.gz`` file. This greatly speeds
up mesh creation if the same mesh needs to be loaded again at a later
time. If the ``.xml.gz`` file exists but is older than the ``.geo``
file then the mesh is recreated anyway.

For quick interactive inspection of the mesh, it can be plotted from
within Python:

.. code-block:: python

    import dolfin
    dolfin.plot(mesh, interactive=True)


Equivalently to the method just described, the mesh can also be
created directly (without the need of a ``.geo`` file) by passing
the CSG description as a string:

.. code-block:: python

    from finmag.utils.meshes import from_csg
    import textwrap

    csg_string = textwrap.dedent("""\
        algebraic3d
        solid main = sphere ( 0, 0, 0; 1.0 ) -maxh = 0.2;
        tlo main;""")

    mesh = from_csg(csg_string)
