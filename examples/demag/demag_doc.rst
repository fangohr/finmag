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

Demag energy on the unit sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We want to find the demagnetisation field and energy on a unit sphere. The initial magnetisation points in x-direction, i.e. m = (1, 0, 0). The saturation magnetisation is :math:`10^5`. To compute this, simply write

.. code-block:: python

    mesh = UnitSphere(10)
    llg = LLG(mesh)
    llg.set_m((1, 0, 0))
    llg.Ms = 1e5
    llg.setup(use_demag=True)

The demagnetisation energy is defined as

.. math::

    E_\mathrm{demag} = -\frac12 \mu_0 \int_\Omega
    H_\mathrm{demag} \cdot M \mathrm{d}x

Because :math:`H_\mathrm{demag} = -1/3 M` [need reference], and with our definition of m, we can
write

.. math::

    E_\mathrm{demag}
    &= -\frac12 \mu_0 \int_\Omega H_\mathrm{demag} \cdot M \mathrm{d}x \\
    &= -\frac12 \mu_0 \int_\Omega (-\frac13 M) \cdot M \mathrm{d}x \\
    &= \frac16 \mu_0 M_s^2 \int_\Omega m \cdot m \mathrm{d}x \\
    &= \frac16 \mu_0 M_s^2 \int_\Omega 1 \mathrm{d}x \\
    &= \frac16 \mu_0 M_s^2 V,

where :math:`V` indicates the volume of the mesh. With a unit sphere mesh, this yields :math:`V = \frac{4
\pi}{3}`. The magnetic constant is defined as :math:`\mu_0 = 4 \pi 10^{-7}`. Hence, the analytical solution
of our case is

.. math::

    E_\mathrm{demag} = \frac16 \mu_0 M_s^2 V = \frac16 4 \pi 10^{-7} \frac{4\pi}{3} (10^5)^2 = 8772.98

Our implementation on a coarse unit sphere mesh with 10 cells in each direction, gives the energy

.. code-block:: none

    Demag energy: 8610.63631312

The reason for the difference from the analytical solution is mainly due to the coarseness of the
mesh, and because Dolfin produces sphere meshes of very low quality. Our results convert towards the
analytical solution with finer meshes. Running the same simulation on a netgen created sphere mesh
with more than 35,000 vertices, gives

.. code-block:: none

    Demag energy: 8758.92651323

Complete code:

.. literalinclude:: ../examples/demag/test_energy.py


Demag field in uniformly magnetised sphere
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Changing the unit sphere mesh to a sphere of radius 10nm, and :math:`M_s` to :math:`10^6`, gives the same case as the first nmag
`example <http://nmag.soton.ac.uk/nmag/0.2/manual/html/example1/doc.html>`_. Remember that we expect
the demag field to be :math:`-1/3 M`, i.e. :math:`(-10^6/3, 0, 0)`. Our implementation gives

.. literalinclude:: ../examples/demag/results_field.txt

The complete code follows

.. literalinclude:: ../examples/demag/test_field.py

Comparing with nmag
^^^^^^^^^^^^^^^^^^^

Comparing with nmag, the following log-log plot shows the development of the standard deviation for increasingly finer meshes.

.. figure:: ../examples/demag/stddev_loglog.png
    :scale: 75

The development of the x-values, which should converge against -1/3, can be seen here:

.. figure:: ../examples/demag/xvalues.png
    :scale: 75

The `errornorm <http://fenicsproject.org/documentation/dolfin/1.0.0/python/programmers-reference/fem/norms/errornorm.html#dolfin.fem.norms.errornorm>`_ is decreasing as the mesh gets finer.

.. figure:: ../examples/demag/errnorm_loglog.png
    :scale: 75

.. note::

    * TODO: When the building of the boundary element matrix is faster, we can increase the number of vertices.

For an example where we also include the exchange field, please see the exchange-demag example in the next section.

.. rubric:: References

.. [#Knittel] Andreas Knittel, *Micromagnetic simulations of three dimensional core-shell nanostructures*, PhD Thesis, University of Southampton, UK, 2011
