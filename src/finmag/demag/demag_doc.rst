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

The Fredkin-Koehler approach
----------------------------

The idea of the Fredkin-Koehler approach is to split the magnetic potential into two parts, :math:`\phi = \phi_1
+ \phi_2`. :math:`\phi_1` solves the inhomogeneous Neumann problem

.. math::

    \Delta \phi_1 = \nabla \cdot \vec M(\vec r), \quad \vec r \in \Omega,

with

.. math::

    \frac{\partial \phi_1}{\partial \vec n} = \vec n \cdot \vec M

on :math:`\Gamma`. In addition, :math:`\phi_1(\vec r) = 0` for :math:`\vec r \not \in \Omega`. To avoid numerical instability, 
we add a small :math:`\epsilon` multiplied with :math:`\phi` to the inner problem, i.e.

.. math::

    \Delta \phi_1 + \epsilon \phi_1 = \nabla \cdot \vec M(\vec r), \quad \vec r \in \Omega

Multiplying with a test function, :math:`v`, and integrate over the domain, the corresponding variational problem reads

.. math::

    \int_\Omega \nabla \phi_1 \cdot \nabla v - \epsilon \phi_1 \cdot v \mathrm{d}x   = \int_\Gamma (\vec n \cdot \vec M)v \mathrm{d}s - \int_\Omega (\nabla \cdot \vec M)v \mathrm{d}x 

This is carried out by the function compute_phi1() in the FemBemFKSolver class. The code simply reads

.. code-block:: python

    # Define functions
    u = TrialFunction(V)
    v = TestFunction(V)
    n = FacetNormal(V.mesh())
            
    # Define forms
    eps = 1e-8
    a = dot(grad(u),grad(v))*dx - dot(eps*u,v)*dx 
    f = dot(n, M)*v*ds - div(M)*v*dx

    # Solve for the DOFs in phi1
    solve(a == f, self.phi1)

Correspondingly, the equations for :math:`\phi_2`, read

.. math::
    \Delta \phi_2(\vec r) = \left\{ \begin{array}{ll}  0 & \hbox{for } \vec r
    \in \Omega \\
    \phi_1 & \hbox{for } \vec r  \in \Gamma
    \end{array} \right.,

and it disappears at infinity, i.e.

.. math::

    \phi_2(\vec r) \rightarrow 0 \quad \mathrm{for} \quad \lvert \vec r \rvert \rightarrow \infty.

In contrast to the Poisson equation for :math:`\phi_1` which is solved straight forward in a finite domain, we now need to apply a BEM technique to solve the equations for :math:`\phi_2`. First, we solve the equation on the boundary. By eq. (2.51) in [#Knittel]_, this yieds

.. math::

    \Phi_2 = \mathbf{B} \cdot \Phi_1,

with the elements of the boundary element matrix :math:`\mathbf{B}` given by

.. math::

    B_{ij} = \frac{1}{4\pi}\int_{\Gamma_j} \psi_j(\vec r) \frac{(\vec R_i - \vec r) \cdot n(\vec r)}{\lvert \vec R_i - \vec r \rvert^3} \mathrm{d}s + \left(\frac{\Omega(\vec R_i)}{4\pi} - 1 \right) \delta_{ij}.

Here, :math:`\psi` is a set of basis functions and :math:`\Omega(\vec R)` denotes the solid angle. For the solid angle, we have used something that is completely wrong due to ridiculous notation in IEEE 2008 [#IEEE]_, which the author read as

.. math::

    \Omega(\vec R_i) = \int_\Gamma \frac{(\vec R_i - \vec r) \cdot n(\vec r)}{\lvert \vec R_i - \vec r \rvert^3} \mathrm{d}s.

This should of course be fixed as soon as possible, perhaps by using eq. (21) from the same paper. :math:`\vec R_i` are the coordinates of the nodal points, which we have stored in a dictionary with the node numbers as keys and their coordinates as values. The rest is conviniently handled by the Dolfin Expression class. The complete code for computing the demoninator, :math:`\lvert \vec R_i - \vec r \rvert^3`, for one of the nodes with R as coordinates, reads
 
.. code-block:: python

    def __BEMdenominator(self, R):
        """
        Compute the denominator of the fraction in
        the first term in Knittel (2.52)

        """
        w = "pow(1.0/sqrt("
        dim = len(R)
        for i in range(dim):
            w += "(R%d - x[%d])*(R%d - x[%d])" % (i, i, i, i)
            if not i == dim-1:
                w += "+"
        w += "), 3)"

        kwargs = {"R" + str(i): R[i] for i in range(dim)}
        return Expression(w, kwargs)
 
The numerator is computed in the same way. With v and w denoting the numerator and denominator, respectively, we can compute the integral by

.. code-block:: python

    psi = TestFunction(V)
    L = 1.0/(4*DOLFIN_PI)*psi*v*w*ds
 
This has to be restricted to the boundary, which is accomplished by the restrict_to() function from the FemBemDeMagSolver base class. Computing the second term containing the solid angle is done in the exact same way, except this is just a scalar, so we don't multiply it with the basis function. 

Calculating the dot product between :math:`\Phi_1` restricted to the boundary and the boundary element matrix, gives the values of :math:`\Phi_2` on the boundary. These values are then used as essential boundary conditions when solving the Laplace equation for :math:`\phi_2` inside the domain. This is in turn solved by the function solve_laplace_inside() in the solver base class.

.. code-block:: python

    def solve_laplace_inside(self,function):
        """Take a functions boundary data as a dirichlet BC and 
	    solve a laplace equation"""
        V = function.function_space()
        bc = DirichletBC(V,function, "on_boundary")
        u = TrialFunction(V)
        v = TestFunction(V)

        #Laplace forms
        a = inner(grad(u),grad(v))*dx
        A = assemble(a)

        #RHS = 0
        f = Function(V).vector()

        #Apply BC
        bc.apply(A,f)

        solve(A,function.vector(),f)
        return function
 
The scalar magnetic potential is then found simply by adding the contributions, :math:`\phi = \phi_1 + \phi_2`, and the demag field is its negative gradient, :math:`\vec H_{\mathrm{demag}} = - \nabla \phi`.


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


Structure
---------

We have two hierarchies, one for the problems and one for the solvers. The super solver class DeMagSolver takes a problem instance as input, so we will start with the problems.
The super problem class DeMagProblem takes a mesh and the initial magnetisation as input to its constructor. The class FemBemDeMagProblem extends this class, but does not do anything else at the moment.
For each of the test cases we construct a class, which defines its mesh and initial magnetisation in the constructor and then initialises its parent class.

The super solver class DeMagSolver, takes a problem instance as input argument. Then, it interpolates the problem's magnetisation onto a Dolfin VectorFunctionSpace with Discontinouos Galerkin elements. Why we do this, I have no idea. TODO: Find out why we're not using CG. 
It also contains some usefull functions, like getdemagfield which returns the negative gradient, and savefunction which saves its input function in a pvd file.

The class FemBemDeMagSolver contains all methods that are shared between the FK and GCR solvers. This includes the solvelaplaceinside function and a function for restricting a vector to the boundary nodes, as well as creating the dictionary with nodal coordinates. At the moment, we have defined the function space in which the magnetic potentials are defined on CR elements. These elements have their nodes on each edge midpoint instead of in the corners of the triangle for CG elements. This is to avoid singularity when computing the boundary element matrices.

Both the FK and GCR solvers extend this class, in addition to a FK and GCR super class where the :math:`\phi_1` and :math:`\phi_a` potentials are computed respectively. Then their method solve, for both solvers, compute the first potential using standard FEM, the second potential on the boundary using BEM, the Laplace equation for the second potential inside the domain, and then return the sum of the two potentials.


Examples
--------

The unit sphere, unit circle and unit interval are already implemented in the file prob_fembem_testcases. To compute the magnetic potential on e.g. a unit sphere using the Fredkin-Koehler approach, simply write

.. code-block:: python

    problem = MagUnitSphere()
    solver = FemBemFKSolver(problem)
    phi = solver.solve()
 
The same applies for the Garcia-Cervera-Roma approach, the only difference is the name of the solver class.
When :math:`\phi` is computed using the solvers solve() function, the demag field :math:`\vec H_{\mathrm{demag}} = - \nabla \phi(\vec r)` can be obtained by

.. code-block:: python

    phi = solver.solve()
    H_demag = solver.get_demagfield(phi)
 
The following example shows how to create a problem with a mesh stored in a file and user-provided initial magnetisation, solve the problem using the GCR solver, and obtain the demag field.

.. code-block:: python

    mesh = Mesh("sphere.xml")
    M = ("1.0", "0.0", "0.0")
    problem = FemBemDeMagProblem(mesh, M)
    solver = GCRDeMagSolver(problem)
    phi = solver.solve()
    H_demag = solver.get_demagfield(phi)
 
The magnetic potential and the demag field can now be saved to pvd files.

.. code-block:: python

    solver.save_function(phi, "GCR_phi")
    solver.save_function(H_demag, "GCR_demagfield")
 


Further work
------------

* Optimize code, avoid loops

* Extend the solvers to save and load the boundary element matrix to/from a numpy file.

* We should at some point project the demag field onto a CG vector function space in order to be able to add it to the other fields.
 
* get_demagfield() should perhaps be able to return the demag field without having to receive phi as an argument. I suggest something like

.. code-block:: python

    solver.solve()
    H_demag = solver.get_demagfield()
 
 
* A really import thing is to extend the DeMagSolver class to handle magnetisation not only as a string which is given to Dolfin Expression, but it should handle Dolfin Function instances as well. I guess it will be initiated from the LLG class, so it should be able to use the same format for M as used by LLG.

* Include some results and pictures in this file.


.. rubric:: References

.. [#Knittel] Andreas Knittel, *Micromagnetic simulations of three dimensional core-shell nanostructures*, PhD Thesis, University of Southampton, UK, 2011

.. [#IEEE] Massimo Fabbri, *Magnetic Flux Density and Vector Potential of Uniform Polyhedral Sources*, IEEE Transactions on Magnetics, 2008


