Example 'Computing the Time Development of a System' from Nmag
==============================================================

This example corresponds to the one with the same name in nmag's manual. It
computes the time development of the magnetisation in a bar with (x, y, z)
dimensions 30 nm x 30 nm x 100 nm. The initial magnetisation is pointing in
the [1, 0, 1] direction, i.e. 45 degrees away from the x axis in the direction
of the (long) z-axis.

First of all, the mesh defined in the file `bar.geo` needs to be loaded into
FinMag. This is done with

.. literalinclude:: ../examples/nmag_example_2/run_finmag.py
    :lines: 10

Once we have a mesh, the simulation can be set up:

.. literalinclude:: ../examples/nmag_example_2/run_finmag.py
    :lines: 12-20

A simulation object is created from the mesh, the initial magnetisation is set
and the exchange interaction and demagnetising field are added to the effective
field. The simulation is told to save the averaged magnetisation every 5 ps
and run for 300 ps.

The plot shows the good agreement between finmag's and nmag's results.

.. image:: ../examples/nmag_example_2/agreement_averages.png
    :align: center
