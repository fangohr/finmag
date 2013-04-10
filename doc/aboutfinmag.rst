The Finmag project
==================

Most of the information in this chapter is only relevant to developers of the code. 

Overview code layout 
--------------------

We combine Python code using Dolfin from the Fenics project, with some C++ code (called "native" here) for performance reasons where necessary. Furthermore, a Boost-Python wrapper for the CVODE algorithm from Sundials is used for time integration.

Python class layout
^^^^^^^^^^^^^^^^^^^

The :doc:`LLG <modules/LLG>` class contains the physics of the LLG equation, and the central dolfin function that carries the normalised magnetisation (currently in LLG._m) which is used in the the :doc:`Exchange <modules/Exchange>`, :doc:`Anisotropy <modules/Anisotropy>`, :doc:`Demag <modules/FKDemag>` and :doc:`DMI <modules/DMI>` class to compute the respective fields and energies.
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

Customisation via ``.finmagrc``
-------------------------------

There is (currently rather rudimentary) support for customisation of Finmag's behaviour via a configuration file. Either of the files ``~/.finmagrc`` and ``~/.finmag/finmagrc`` will be taken into account (if they exist). This is an example of a minimalistic sample configuration file:

.. literalinclude:: finmagrc_template.txt

Configuration options are collected in sections (such as 'logging' in the example above), and each section contains assignments for the variables to be configured. Currently (as of 4.10.2012) the configurable options are as follows:

* Section ``logging``:

  - ``logfile``: Specifies one or multiple logfiles to which the Finmag output will be copied. Any non-existing files will be created. If a file already exists then new output will be appended at the end.

    Filenames specified by relative paths are interpreted as being relative to the directory in which the simulation was started. This is useful if one wants to keep logging output separate for each simulation. If, on the other hand, a global logfile is desired (in which all output from all simulations is kept together) then an absolute filename should be given.

    It is possible to specify multiple logfiles, as in the example above. In this case they should appear one per line, and each such line must begin with at least one space.

  - ``console_logging_level``: Sets the threshold logging level for logging messages that are sent to console using the Python `logging <http://docs.python.org/library/logging.html>`__ library. Default value is DEBUG; set it to WARN or higher to reduce the amount of messages sent to the console. Possible values are DEBUG, INFO, WARN, ERROR, CRITICAL.

  - ``dolfin_logging_level``: Sets the threshold logging level for messages from FEniCS/Dolfin. Dolfin has its own logging facility for C++ code that is separate from Python's `logging <http://docs.python.org/library/logging.html>`__. Possible values are DEBUG, INFO, WARN, ERROR, CRITICAL, PROGRESS.

  - ``color_scheme``: Selects a color scheme for the logger. Allowed values are 'dark_bg' and 'light_bg' (for terminals with dark and light background, respectively), or 'none' to supress colouring.

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


A note on parallel execution
----------------------------

**FEniCS** supports parallel execution in different ways:

* backends such as PETSC allow to use MPI to deal with linear algebra tasks in parallel (either across multiple machines, or 'abusing' MPI across multiple cores on the same host)

* OpenMP (to exploit multiple cores on the same CPU, or at least the same node [there may be more than one CPU per node]) is partially used, for example for the 'assemble' commands. For studies where the assemble command is called ones, this may be of moderate value, but can be significant if matrices/vectors need to be recomputed.

* FEniCS Python code can be written so that it runs in parallel. Often this seems to be automatic, but at times one needs to be careful. We have so far (Sept 2012) not paid any attention to this in the Finmag project.

This information is based on Hans' vague memory of reading the FEniCS documentation late in 2011. 

**Sundials/CVODE** is conceptually the point that is ''most serial'' in Finmag: while in principle one can use CVODE in parallel fashion, this needs a parallel implementation of the nvector data class. For now, we use such a class which connects to numpy's array object, so that the data vector that cvode operators on is the same as that of a numpy array. This is not the same vector (in memory) as the dolfin field, so we need to copy this backwards and forwards when required (i.e. every time the right-hand-side of our ODE needs to be evaluated). While the dolfin vector depends on the backend (and in the case of the PETSC backend could in principle be distributed across multiple nodes), the numpy array object is not parallised in that way. 

**OpenMP** in our own (BOOST.Python)-linked code is used where possbile, i.e. multiple cores are used where available, for example in the computation of the BEM for the Fredkin Koehler method.

**GPUs**: there are recent efforts to provide a GPU based backend. Anders Johanson has tried to use that with Finmag, but could only achieve very modest speed ups (30%, where we were hopeing for numbers around 1000%, i.e. 10 time faster). The current thinking is that the linear algebra operations are based on iterative methods which involve mostly reading of data (and then a little numerical operation) and thus can not fully exploit the power of the many cores on GPUs, but instead are limited by the memory bandwidth).

**Summary**: We run finmag at the moment as a serial program; and multiple cores are exploited automatically. This can certainly be improved. However, going beyond use of one node (which means: use of MPI) requires revising the Python code, and will then not be ideal as the sundials interface is serial (so all the data for the state vector would have to be gathered to the master node, then given to CVODE, and for the computation of the effective fields which contribute to the right-hand-side, scattered across all MPI nodes). The first point (revising FEniCS Python code for parrallel execution) may not be too bad, and the second point ('serial' interface to cvode through numpy array) may not slow things down that much: once the right-hand side is computed, we can use FEniCS' PETSC backend (which exploits MPI).  [On the other hand, with increasing core numbers, it may be sufficient to focus on speed up through multicore (i.e. OpenMP) techniques.]


