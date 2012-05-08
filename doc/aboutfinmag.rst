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

.. include:: install.rstext


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

Comparison with other micromagnetic packages
--------------------------------------------

:math:`\renewcommand{\subn}[2]{#1_{\mathrm{#2}}}`

To monitor the correct operation of finmag, automatic tests are set up
to compare finmag's simulation results to other micromagnetic packages. These
are `nmag <http://nmag.soton.ac.uk/nmag/>`_, `oommf <http://math.nist.gov/oommf/>`_
and `magpar <http://www.magpar.net>`_.  After the same problem specification
has been written for the four packages,
the simulations are run and the results stored.

Because magpar and nmag both use the finite element method, results
defined on the same mesh can be compared easily to finmag. Oommf uses the finite
difference method and uses meshes built of cubic cells. To check a field
computed with finmag against its oommf equivalent, the former is probed at
the locations of the vertices of the corresponding oommf mesh. When fields are
compared beetween finmag and nmag, special care has to be taken: Instead of
the field, the cross product of the magnetisation and the
field :math:`m \times H` has to be evaluated. Nmag takes the liberty of having
additional contributions in their vectors, because only the perpendicular parts
enter the LLG equation anyways.

This yields a new vector which can be used for the comparison.

The relative difference between
finmag's results and another package is computed with this formula:

.. math::

    \Delta = \frac{\subn{\vec{r}}{finmag} - \subn{\vec{r}}{ref}}{\max(\|\subn{\vec{r}}{ref}\|)} 

The absolute difference is divided by the maximum value of the euclidean norm
of the vectors in the reference simulation to filter out big relative errors
due to comparing values which are supposed to be zero.

The maximum relative difference is :math:`\subn{\Delta}{max}` and the mean is
:math:`\bar{\Delta}`. The standard deviation is the square root of the average
of the squared deviations from the mean and denoted by :math:`\sigma`.
Usually, :math:`\subn{\Delta}{max}` is rounded up to one decimal place and used
as a safeguard against programming mistakes, called :math:`\subn{\Delta}{test}`.
By comparing the hard-coded :math:`\subn{\Delta}{test}` to the
computed :math:`\subn{\Delta}{max}` and displaying an error message if
:math:`\subn{\Delta}{max} > \subn{\Delta}{test}`, regressions can be identified.

The Exchange Interaction
^^^^^^^^^^^^^^^^^^^^^^^^

For the comparison of the exchange field, it is computed on a one-dimensional mesh
with a starting magnetisation as described by

.. math::
    
    m_x = ( 2x - 1 ) \cdot \frac{2}{3} 

    m_y = \sqrt{1 - m_x^2 - m_z^2} 

    m_z = \sin(2 \pi x) \cdot \frac{1}{2}

where :math:`x \in [0; 1]`.

The values for the relative difference are listed in :ref:`exchange_table`.
Note that the data in the table is re-created on the fly when compiling
the documentation.

.. include:: ../src/finmag/tests/comparison/exchange/table.rst

Because this problem is defined on a one-dimensional mesh, no comparison with
magpar is possible. However ``src/finmag/tests/magpar/test_exchange_compare_magpar.py``
is run with :math:`\subn{\Delta}{test} = 9\times 10^{-9}`.

Uniaxial Anisotropy
^^^^^^^^^^^^^^^^^^^

The initial magnetisation used for the computation of the anisotropy field
is defined by

.. math::

    m_x = ( 2 - y ) \cdot ( 2x - 1) \cdot \frac{1}{4} 

    m_y = \sqrt{1 - m_x^2 - m_z^2} 

    m_x = ( 2 - y ) \cdot ( 2z - 1) \cdot \frac{1}{4} 

where :math:`x, y, z \in [0; 1]`.

The values for the relative difference are listed in :ref:`anis_table`.

.. include:: ../src/finmag/tests/comparison/anisotropy/table.rst

The Demagnetising field
^^^^^^^^^^^^^^^^^^^^^^^

A sphere was magnetised in the :math:`(1, 0, 0)` direction and the demagnetising
field computed with finmag, nmag and magpar. Those three fields were
additionally compared to the analytical solution :math:`(-\frac{1}{3}, 0, 0)`.

The values for the relative difference are listed in :ref:`demag_table`. It is
worth noting that the nmag and finmag solution are close to each other, with
magpar further away both from them and the analytical solution.

.. include:: ../src/finmag/tests/comparison/demag/table.rst

Solution of the LLG equation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A comparison of our solution of the LLG equation with an analytical model is done
in the section :ref:`macrospin_example`. For completeness, here is a comparison of our
results which those of oommf, for a homogeneous effective field and starting
magnetisation as decribed in ``finmag/tests/comparison/test_dmdt.py``.

+---------+------------------------------+----------------------------+-----------------------------+-----------------------------+
|         |  :math:`\subn{\Delta}{test}` |:math:`\subn{\Delta}{max}`  | :math:`\bar{\Delta}`        | :math:`\sigma`              |
+=========+==============================+============================+=============================+=============================+
| oommf   |  :math:`3\times 10^{-16}`    |:math:`2.28\times 10^{-16}` | :math:`1.12\times 10^{-16}` | :math:`3.86\times 10^{-17}` |
+---------+------------------------------+----------------------------+-----------------------------+-----------------------------+
