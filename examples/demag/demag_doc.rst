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

We have currently implemented two different approaches for solving the demagnetisation field, the :doc:`Fredkin-Koehler method <modules/FKSolver>` and the Garcia-Cervera-Roma method.

.. note::

    TODO: Move documentation to the GCR Solver and link to it in the same way as for the FK Solver.

The Garcia-Cervera-Roma approach
--------------------------------

This approach is similar to the Fredkin-Koehler approach, so we will just comment on the differences between the approaches. As before, the magnetic scalar potential is diveded into two parts, :math:`\phi = \phi_a + \phi_b`, but the definition of these are different. :math:`\phi_a` is the solution of the inhomogeneous Dirichlet problem defined as

.. math::

    \Delta \phi_a(\vec r) = \nabla \vec M(\vec r)

inside the domain, and

.. math::

    \phi_a(\vec r) = 0

on the boundary and outside the domain. This is solved in a similar manner as before, with the variational forms and boundary condition given by

.. code-block:: python

    #Define forms
    a = dot(grad(u),grad(v))*dx
    f = (-div(self.M)*v)*dx  #Source term

    #Define Boundary Conditions
    bc = DirichletBC(V,0,"on_boundary")
 
The second potential, :math:`\phi_b`, is the solution of the Laplace equation 

.. math::
    
    \Delta \phi_b = 0

inside the domain, its normal derivative has a discontinuity of

.. math::

    \Delta \left(\frac{\partial \phi_b}{\partial n}\right) = -n \cdot \vec M(\vec r) + \frac{\partial \phi_a}{\partial n}

on the boundary and it vanishes at infinity, with :math:`\phi_b(\vec r) \rightarrow 0` for :math:`\lvert \vec r \rvert \rightarrow \infty`.
As for the Fredkin-Koehler approach, the boundary problem can be solved with BEM. Unlike the Fredkin-Koehler approach, where the vector equation for the second part of the potential is the product between the boundary element matrix and the first potential on the boundary, we now have

.. math::

    \Phi_b = \mathbf{B} \cdot Q.

The vector :math:`Q` contains the potential values :math:`q` at the sites of the surface mesh, with :math:`q` defined as the right hand side of the boundary equation,

.. math::

    q(\vec r) = -n \cdot \vec M(\vec r) + \frac{\partial \phi_a}{\partial n}.

This vector is assembled in the function assembleqvectorexact in the CGRFemBemDeMagSolver class. The values of the boundary element matrix :math:`\mathbf{B}` is given by

.. math::

    B_{ij} = \frac{1}{4\pi}\int_{\Omega_j} \psi_j(\vec r)\frac{1}{\lvert \vec R_i - \vec r \rvert} \mathrm{d}s.

The way this is computed in the GCR solver is practically the same as for the FK solver. Solving the Laplace equation inside the domain and adding the two potentials, is also done in the exact same way as before.

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
