The Finmag project
==================

Overview code layout
--------------------

We combine Python code using Dolfin from the Fenics project, with some C++ code (called "native" here) for performance reasons where necessary. Furthermore, a Boost-Python wrapper for the CVODE algorithm from Sundials is used for time integration.

Python class layout
^^^^^^^^^^^^^^^^^^^

The :doc:`LLG <modules/LLG>` class contains the physics of the LLG equation, and the central dolfin function that carries the normalised magnetisation (currently in LLG._m) which is used in the the :doc:`Exchange <modules/Exchange>`, :doc:`Anisotropy <modules/Anisotropy>`, :doc:`Demag <modules/FKSolver>` and :doc:`DMI <modules/DMI>` class to compute the respective fields and energies.

The ``TimeStepper class`` [XXX exact name, and add link (this is not yet included in the documentation)] is used for time integration, and then carries the state of the independent degrees of freedom (such as the magnetisation) when sundials is used. (The LLG._m is not suitable for this, although it is used by the time integration class internally.)

We plan to have a convenience class at the top level (the ``Simulation class``) which is meant to follow the Nmag simulation Class whereever we feel there is nothing to improve upon the Nmag model. (If there is no good reason to do something different, let's keep it easy and do it the same way.) The documentation for the Nmag Simulation class is `here <http://nmag.soton.ac.uk/nmag/current/manual/html/command_reference.html#simulation>`_ although the practical usage examples in the `tutorial <http://nmag.soton.ac.uk/nmag/current/manual/html/guided_tour.html>`_ are probably more useful to follow.




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

.. include:: ../devtests/gpu/finmag_gpu.txt


Team
----

Team and contact details are available at https://bitbucket.org/fangohr/finmag/wiki/Team_members


Tools
-----

We use the following tools:

* `Bitbucket <https://bitbucket.org/fangohr/finmag>`__ to host the mercurial repository

  This also provides a `Wiki <https://bitbucket.org/fangohr/finmag/wiki/Home>`_ with some useful information.

* `Pivotal tracker <https://www.pivotaltracker.com/projects/475919>`__ to track tasks, bugs, etc.

* `Jenkins <http://summer.kk.soton.ac.uk:8080/job/finmag>`__ to automatically build and test the code and documentation after every commit to the bitbucket repository.

* `IRC <https://bitbucket.org/fangohr/finmag/wiki/IRC>`_ for quick questions during the day (or night).

* The finmag-team mailing list at finmag-team@lists.soton.ac.uk
