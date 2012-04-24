FEM/BEM Demag solvers
=====================

From Andreas Knittel's thesis [#Knittel]_ we have that the relation between the demagnetisation field, :math:`\vec
H_{\mathrm{demag}}`, and the magnetic scalar potential, :math:`\phi(\vec r)`, is given by

.. math::

    \vec H_{\mathrm{demag}} = - \nabla \phi(\vec r).

Here, :math:`\phi` is the solution of a Poisson equation,

.. math::

     \Delta \phi(\vec r) = \nabla \cdot \vec M(\vec r),

where :math:`\vec M` denotes the magnetisation with magnitude :math:`\lvert \vec M \rvert = M_s` for
:math:`\vec r \in \Omega`. The magnetisation vanishes outside :math:`\Omega`, hence the scalar
potential has to solve

.. math::

    \Delta \phi(\vec r) = \left\{ \begin{array}{ll}  \nabla \cdot \vec M(\vec r) & \hbox{for } \vec r
    \in \Omega \\
    0 & \hbox{for } \vec r \not \in \Omega
    \end{array} \right.

The scalar potential is continuous at the boundary :math:`\Gamma` of :math:`\Omega`, i.e.

.. math::

    \phi_{\mathrm{ext}} = \phi_{\mathrm{int}} \quad \hbox{on } \Gamma,

with discontinuous normal derivative,

.. math::

    \frac{\partial \phi_{\mathrm{ext}}}{\partial \vec n} = \frac{\partial
    \phi_{\mathrm{int}}}{\partial \vec n} \quad \hbox{on } \Gamma.

It is also required that the potential is zero at infinity, hence :math:`\phi(\vec r) \rightarrow 0` for :math:`\lvert \vec r \rvert \rightarrow \infty`.

We have currently implemented two different approaches for solving the demagnetisation field, the
:doc:`Fredkin-Koehler method <modules/FKSolver>` and the :doc:`Garcia-Cervera-Roma method
<modules/GCRSolver>`.

Examples
--------

To compute the demagnetisation field on e.g. a unit sphere using the Fredkin-Koehler approach, simply write

.. code-block:: python

    mesh = UnitSphere(10)
    V = VectorFunctionSpace(mesh, "CG", 1)
    m = interpolate(Constant((1,0,0)), V)
    Ms = 1
    solver = FemBemFKSolver(V, m, Ms)
    H_demag = solver.compute_field()

The following example shows how to obtain the demag field for the same case as the first nmag `example <http://nmag.soton.ac.uk/nmag/0.2/manual/html/example1/doc.html>`_.

.. literalinclude:: ../examples/demag/demag_example.py

Comparing with nmag, the following log-log plot shows the development of the standard deviation for increasingly finer meshes.

.. figure:: ../examples/demag/stddev_loglog.png
    :scale: 75

The development of the x-values, which should converge against -1/3, can be seen here:

.. figure:: ../examples/demag/xvalues.png
    :scale: 75

The `errornorm <http://fenicsproject.org/documentation/dolfin/1.0.0/python/programmers-reference/fem/norms/errornorm.html#dolfin.fem.norms.errornorm>`_ is decreasing as the mesh gets finer.

.. figure:: ../examples/demag/errnorm.png
    :scale: 75

.. note::

    * TODO: When the building of the boundary element matrix is faster, we can increase the number of vertices.
    * TODO: Maybe include a log-log plot of the errornorm instead of the "normal" plot.

For a more interesting example where we also include the exchange field, please see the exchange-demag example in the next section.

.. rubric:: References

.. [#Knittel] Andreas Knittel, *Micromagnetic simulations of three dimensional core-shell nanostructures*, PhD Thesis, University of Southampton, UK, 2011
