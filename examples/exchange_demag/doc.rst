Exchange and demagnetisation example
====================================

In this example, we compute the time development of a system which initially points 45 degrees away from the x-axis, in the direction of the z-axis (i.e. initial magnetisation points in the [1,0,1]-direction). We consider both the exchange and demag fields (anisotropy not considered) and compare the results with nmag. We use a time step of size 5.0e-12 and consider 60 time steps.

The geometry we use is a bar of dimensions 30 x 30 x 100 nm. This is stored in the .geo-file bar_30_30_100.geo and is converted automatically to a Dolfin compatible .xml.gz-file using Netgen and the dolfin-convert script. The saturation magnetisation is 0.86e6 (A/m) and the exchange coupling is 13.0e-12 (J/m). The LLG damping constant we use is 0.5.

The finmag code for setting up the LLG object reads

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py
    :lines: 42-48

The time integrator is created by

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py
    :lines: 50,51

and for each time step, t, we call the integrator by

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py
    :lines: 65,66

For now, our relative error from the nmag implementation is below 1e-4. This plot shows the comparison between the finmag and nmag results.

.. image:: ../examples/exchange_demag/exchange_demag.png
    :scale: 75
    :align: center

Time comparisson between the nmag and finmag implementations shows the following:

.. include:: ../examples/exchange_demag/timings/results.rst
    :literal:

The complete code follows.

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py

.. rubric:: Footnotes
