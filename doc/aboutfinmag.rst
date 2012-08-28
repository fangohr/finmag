The Finmag project
==================


Overview code layout 
--------------------

We combine Python code using Dolfin from the Fenics project, with some C++ code (called "native" here) for performance reasons where necessary. Furthermore, a Boost-Python wrapper for the CVODE algorithm from Sundials is used for time integration.

Python class layout
^^^^^^^^^^^^^^^^^^^

The :doc:`LLG <modules/LLG>` class contains the physics of the LLG equation, and the central dolfin function that carries the normalised magnetisation (currently in LLG._m) which is used in the the :doc:`Exchange <modules/Exchange>`, :doc:`Anisotropy <modules/Anisotropy>`, :doc:`Demag <modules/FKSolver>` and :doc:`DMI <modules/DMI>` class to compute the respective fields and energies.
The Exchange, Anisotropy and DMI module are based on the :doc:`EnergyBase <modules/EnergyBase>` class.

The ``TimeStepper class`` [XXX exact name, and add link (this is not yet included in the documentation)] is used for time integration, and then carries the state of the independent degrees of freedom (such as the magnetisation) when sundials is used. (The LLG._m is not suitable for this, although it is used by the time integration class internally.)

We plan to have a convenience class at the top level (the ``Simulation class``) which is meant to follow the Nmag simulation Class whereever we feel there is nothing to improve upon the Nmag model. (If there is no good reason to do something different, let's keep it easy and do it the same way.) The documentation for the Nmag Simulation class is `here <http://nmag.soton.ac.uk/nmag/current/manual/html/command_reference.html#simulation>`_ although the practical usage examples in the `tutorial <http://nmag.soton.ac.uk/nmag/current/manual/html/guided_tour.html>`_ are probably more useful to follow.


Important data structures to store fields
-----------------------------------------

The central field for storing the magnetisation in dolfin is a ``dolfin.Function`` object. This has some references to a VectorSpaceFunction (and thus a mesh, and basis function types and order) and a vector that keeps the coefficients of the functions on the vertices. For a order 1 basis function, the vertices are the same as the mesh nodes. For higher order, there will be more vertices along the edges, surfaces or volumina.

The vector format that stores the coefficients can be seen as the state vector for the magnetisation. In dolfin, the vector class is an abstract class, that can be used with a number of backends. By default, we use the Petsc backend, which goes well together with storing the operator matrices as (sparse) Petsc matrices. However, other backends can be chosen if desired (for example for GPU calculations).

We thus have one ``Dolfin.Function object`` for the magnetisation.

We can get access to the coefficients in numpy-array format using the ``.array()`` method of the ``Dolfin.Function`` object.

For the time integration, we give a ``numpy.array`` to sundials, which carries out the time integration, and then updates that numpy.array with the magnetisation coefficients at the requested time.

Once this is done, we need to copy the coefficients from the numpy array, back into the Dolfin-Function object (and also at intermediate times, whenever sundials requests an updated calculation of the effective field, i.e. when the right hand side of the ODE system needs to be evaluated).


These are the two basic ways of storing the magnetisation vector in memory: the ``dolfin.Function`` object that we need to solve the PDEs (at a given time), and the numpy.array that we need for the sundials-driven time integration. There seems no way around having these two, and copying data from one to the other is undesirable, but not a huge burden in comparison to the actual calculations.

In addition to these two fundamental data types, we may want to have different *views* (in numpy terminology) onto the same data: as the magnetisation vector has 3N entries for N magnetisation vectors, there is a question over how we arrange the data.

By default, we follow the Dolfin orientation of storing first the N entries for the x-component of the magnetisation, then N entries for the y-component, and finally N entries for the z component. This is stored in a chunk of 3N elements in memory (typically one element is double floating point type that uses 8 bytes) in this order.

At times, we would rather look at this as N 3d vectors. Numpy supports views in varies ways -- if used correctly, the data *looks* like having a different shape, whereas actually the same data is used, but numpy provides some clever interface (the ``view``) onto it.

We should check at some point that where we create such views repeatedly, these are actually views and not copies of the data.

Details on the interaction with cvode (Sundials)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the interaction with sundials' cvode integrator, we have written an implementation of the nvector data type that provides the interface functions required by cvode and our implementation uses the numpy.array data structure internally to store the vector of degrees of freedom to integrate. This is good as the data does not need to be copied from the numpy.array that we use in the finmag code to a cvode-specific data structure when the carry out the time integration using cvode. This also means that the cvode integrator can do operations on that data vector -- of which there may be many per evaluation of the right hand side -- without requiring extra copies of the data from one data structure to another.





Download
--------

The source is available from bitbucket. To clone use::

  hg clone ssh://hg@bitbucket.org/fangohr/finmag

or::

  hg clone https://fangohr@bitbucket.org/fangohr/finmag

Installation
------------

.. include:: install.txt


Finmag and GPU
--------------

.. include:: ../sandbox/gpu/finmag_gpu.txt


Team
----

Team and contact details are available at https://bitbucket.org/fangohr/finmag/wiki/Team_members


Tools
-----

We use the following tools:

* `Bitbucket <https://bitbucket.org/fangohr/finmag>`__ to host the mercurial repository

  This also provides a `Wiki <https://bitbucket.org/fangohr/finmag/wiki/Home>`_ with some useful information.

  To get email notifications of various events on bitbucket (including people adding comments to your commits), please become a 'follower' of the project. (Probably clicking on the heart labelled 'follow' will do this.)

* `Pivotal tracker <https://www.pivotaltracker.com/projects/475919>`__ to track tasks, bugs, etc.

* `Jenkins <http://summer.kk.soton.ac.uk:8080/job/finmag>`__ to automatically build and test the code and documentation after every commit to the bitbucket repository.

* `IRC <https://bitbucket.org/fangohr/finmag/wiki/IRC>`_ for quick questions during the day (or night).

* The finmag-team mailing list at finmag-team@lists.soton.ac.uk

Overview repository structure
-----------------------------

- ``background`` - background info that is crucial for the code, for example key publications. Add (binary) data (such as pdf) sparingly to keep repository size small.
- ``bin``: - scripts we need; currently only used by Jenkins
- ``sandbox``: a directory used to develop new features, and to have a repository for code that is incomplete, and thus would fail regression tests. When a new feature is completed, it (and associated tests) should be move to the ``src`` directory (and thus execute automatically). One should also add an example of usage into the ``examples`` directory which will be included in the documentation.
- ``sandbox/basics-*``: ``sandbox`` is also sometimes used to test small things that may be worth storing in the repository (because we need to get back to it and/or it could be useful to other developers) but which is not really worth including in the finmag package): typically to improve our understanding of some software method.
- ``doc``: the sphinx-based documentation. This is automatically built by jenkins after every push to the bitbucket server, and failures are reported as warnings. Makes use of code in ``examples``.
- ``examples``: a gallery of example code that demonstrates particular features. Usually combined with a file documenting the example, which is included in the sphinx-based documentation. The documentation is built on jenkins after ``py.test`` has executed in the ``examples`` directory: this can be used to create bitmaps, data tables for the documentation that should be re-computed with every built.
- ``install``: collection of scripts 
- ``logbooks``: contains a dump of the IRC chats (committed at midnight everyday if activity has taken place) and the script that listens to the IRC chat. (The script runs as user 'fangohr' or 'summer.kk.soton.ac.uk.) The script deamon appears as "DokosBot" in the IRC chat. If he is missing, the script needs to be restarted.
- ``native``: C++ code that is integrated into the finmag python code using Boost.Python
- ``src``: the location of the main finmag package. The ``PYTHONPATH`` needs to point to this directory, so that ``import finmag`` can import the package from ``src/finmag``.

A note on code duplication
--------------------------

The development in ``sandbox`` often copies existing code (such an energy class, llg.py, etc) and modifies it until it works. At that point we have a lot of duplicated code in the system. It would then be desirable to add tests to the new code that test the functionality, and then to refactor the new code to avoid code duplication, and then to move this code from ``sandbox`` into ``src``. Refactoring means to essentially write it again taking into account that we now know what it shoud do and how it can work. The art in particular is to merge it so with the main code (in ``src``) that we avoid code duplication. This may require to modify the existing code (but so that nothing else breaks that relies on it). The long list of tests we have (both for already existing code, and the new feature under question) should help with this.

When a feature has been moved from ``sandbox`` to ``src``, accompanying documentation should be added, typically best done through one or more usage examples to be added to the ``examples`` directory.

A note on ``sandbox``
---------------------

This is a somewhat messy place that serves different purposes, including:

- developing new features
- tutorial-like scripts for developers
- other stuff for internal use that was never completed, or has not yet moved to the src directory (such as the GPU usage)

The name is not ideal either. The good news is that we should be able to re-arrange and rename files and directories in here as nothing (neither the documentation, nor main finmag code under ``src`` should depend on it).



