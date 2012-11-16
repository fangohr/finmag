Investigating the reversal process of a grain
=============================================

This example reproduces parts of the paper "*Reversal Modes, Thermal Stability
and Exchange Length in Perpendicular Recording Media*" from Suess, 2001.

In the simulation, a cylindrical mesh is created with

.. literalinclude:: ../examples/magnetic_grain/suess_2001.py
    :language: python
    :lines: 18

It has a radius of 6 nm, a height of 20 nm and a soft upper bound on mesh
coarseness of 2 nm.

The initial magnetisation is set to point straight up in direction of the
uniaxial anisotropy axis, while an external field is specified to point down
with a misalignment of one degree.

.. literalinclude:: ../examples/magnetic_grain/suess_2001.py
    :language: python
    :lines: 14, 17, 25

The reversal process of the magnetisation in the grain is observed for three
different external field strengths.

.. literalinclude:: ../examples/magnetic_grain/suess_2001.py
    :language: python
    :lines: 10, 13, 15-17, 22, 23, 29

The average polarisation in z-direction is plotted as a function of time until
it reaches a value of -0.5:

.. literalinclude:: ../examples/magnetic_grain/suess_2001.py
    :language: python
    :lines: 34, 37-41, 45-48

and shown below. It matches figure 4 of the cited paper.

.. image:: ../examples/magnetic_grain/mz.png
