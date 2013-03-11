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


Usage
-----

Usually, it's sufficient to create a Demag object and give it as an argument to the Simulaton object, e.g.

.. code-block:: python

    >>> sim = Simulation(mesh)
    >>> sim.add(Demag())

The Fredkin-Koehler method ("FK") is the default algorithm. If we instead want to use the Garcia-Cervera-Roma approach ("GCR"), we simply need to give this as argument.

.. code-block:: python

    >>> sim = Simulation(mesh)
    >>> sim.add(Demag("GCR"))

For both methods, we have two linear solvers. One for the poisson problem and one for the laplace problem, and we use Krylov solvers to solve both of them. There are a great amount of different methods and preconditioners available, dependent on installation. To list all of them, use

.. code-block:: python

    >>> list_krylov_solver_methods()
    >>> list_krylov_solver_preconditioners()

In order to run a benchmark test of all possible krylov methods and solvers 
it is enough to create a Demag object with the kwarg "bench" = True.

.. code-block:: python

    >>> demag = Demag(bench = True)
    >>> demag.setup(V, m, Ms, unit_length)
    >>> demag.compute_field()


We use "default" as default method and preconditioner. This can be changed after the demag object is created.

.. code-block:: python

    >>> demag = Demag()
    >>> demag.parameters["poisson_solver"]["method"] = "cg"
    >>> demag.parameters["poisson_solver"]["preconditioner"] = "ilu"
    >>> demag.parameters["laplace_solver"]["method"] = "cg"
    >>> demag.parameters["laplace_solver"]["preconditioner"] = "ilu"

Here, we enable the solver to use the conjugate gradient method with incomplete LU factorization for both the poisson problem and the laplace problem.

As the Krylov solvers are created in the constructor, we have the opportunity to change all default parameters, e.g. tolerances or maximum number of iterations, before the solving starts. The default values can be found e.g.

.. code-block:: python

    >>> s = KrylovSolver()
    >>> p = s.parameters
    >>> print p.to_dict()["relative_tolerance"]
	1e-06


To set e.g. the relative tolerance for the poisson solver to 1e-10, the syntax reads

.. code-block:: python

    >>> demag = Demag()
    >>> demag.poisson_solver.parameters["relative_tolerance"] = 1e-10
    >>> sim.add(demag)

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

.. literalinclude:: ../examples/demag/demagenergies.txt 


The reason for the difference from the analytical solution is mainly due to the coarseness of the
mesh, and because Dolfin produces sphere meshes of very low quality. Our results convert towards the
analytical solution with finer meshes. Running the same simulation on a netgen created sphere mesh
with more than 35,000 vertices, gives

.. code-block:: none

    FK Demag energy: 8758.92651323

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

Comparing nmag, Finmag FK and Finmag GCR
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this section the results from the nmag demag solver are compared to those of the Finmag FK
:doc:`Fredkin-Koehler method <modules/FKSolver>` and Finmag GCR :doc:`Garcia-Cervera-Roma method
<modules/GCRSolver>`. Additionally the GCR method is tested with two different methods of q vector assembly, the default
point evaluation method, and the box method.

The following log-log plot shows the development of the standard deviation for increasingly finer meshes.

.. figure:: ../examples/demag/stddev_loglog.png
    :scale: 75

The development of the x-values, which should converge against -1/3, can be seen here. As nmag uses
the FK method for demag computation as well, the results are quite similar to the finmag FK method:

.. figure:: ../examples/demag/xvalues.png
    :scale: 75

Comparing the GCR with nmag gives the following results

..
    .. figure:: ../examples/demag/xvaluesgcr.png
        :scale: 75

[This plot is currently de-activated because the GCR solver doesn't work with dolfin-1.1]

The `errornorm <http://fenicsproject.org/documentation/dolfin/1.0.0/python/programmers-reference/fem/norms/errornorm.html#dolfin.fem.norms.errornorm>`_
is decreasing as the mesh gets finer. The precision of the GCR method with point evaluation q assembly is better than the GCR method with the box method.  

.. figure:: ../examples/demag/errnorm_loglog.png
    :scale: 75

The bem assembly time is plotted here, increasing with the number of vertices.

.. figure:: ../examples/demag/bemtimings.png
    :scale: 75

The runtime minus bem assembly is plotted here. The FK method is the fastest. The GCR method with point evaluation is slower
than GCR with box method. The point evaluation method,  being implemented in python could be greatly sped up
with a C++ implementation.

.. figure:: ../examples/demag/solvetimings.png
    :scale: 75

The number of Krylov iterations are plotted here.

.. figure:: ../examples/demag/krylovitr.png
    :scale: 75


The linear solver parameters that were used are

.. include:: ../examples/demag/linsolveparams.rst
    :literal:

.. note::

    * Methods and preconditioners which are set to "default" are choosen by the current linear algebra backend of dolfin
      (PetSC, uBLAS, etc...) 

For an example where we also include the exchange field, please see the exchange-demag example in the next section.

.. rubric:: References

.. [#Knittel] Andreas Knittel, *Micromagnetic simulations of three dimensional core-shell nanostructures*, PhD Thesis, University of Southampton, UK, 2011
