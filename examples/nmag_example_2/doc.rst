Computing the Time Development of a System
==========================================

This example corresponds to the one with the same name in nmag's manual. It
computes the time development of the magnetisation in a bar with (x, y, z)
dimensions 30 nm x 30 nm x 100 nm. The initial magnetisation is pointing in
the [1, 0, 1] direction, i.e. 45 degrees away from the x axis in the direction
of the (long) z-axis.

The code for setting up the simulation is shown and then discussed in more detail.

.. literalinclude:: ../examples/nmag_example_2/run_finmag.py
    :lines: 10-20

The plot shows the good agreement between finmag's and nmag's results.

.. image:: ../examples/nmag_example_2/agreement_averages.png
    :align: center
