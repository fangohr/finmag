Exchange and demagnetisation example
====================================

In this example, we compute the time development of a system which initially points 45 degrees away
from the x-axis, in the direction of the z-axis (i.e. initial magnetisation points in the (1,0,1)-direction). We consider both the exchange and demag fields (anisotropy not considered) and compare the results with nmag. We use a time step of size 5.0e-12 and consider 60 time steps.

The geometry we use is a bar of dimensions 30 x 30 x 100 nm. This is stored in the .geo-file bar_30_30_100.geo and is converted automatically to a Dolfin compatible .xml.gz-file using Netgen and the dolfin-convert script. The saturation magnetisation is 0.86e6 (A/m) and the exchange coupling is 13.0e-12 (J/m). The LLG damping constant we use is 0.5.

The finmag code for setting up the LLG object reads

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py
    :lines: 24-30

The time integrator is created by

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py
    :lines: 32,33

and for each time step, t, we call the integrator by

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py
    :lines: 47,48

For now, our relative error from the nmag implementation for the normalized magnetisation is below 1e-4. This plot shows the comparison between the finmag and nmag results.

.. image:: ../examples/exchange_demag/exchange_demag.png
    :scale: 75
    :align: center

For the exchange energy, the relative error of our implementation is also below 1e-4,

.. image:: ../examples/exchange_demag/exchange_energy.png
    :scale: 75
    :align: center

while for the demag energy, the relative error is below 1e-3.

.. image:: ../examples/exchange_demag/demag_energy.png
    :scale: 75
    :align: center

After ten time steps, we compute the energy density through the center of the bar. Comparing the exchange energy density to both nmag and oommf, shows that the finmag implementation
actually is closer to oommf than nmag, because the nmag curve seems to have some noise.

.. image:: ../examples/exchange_demag/exchange_density.png
    :scale: 75
    :align: center

Because of this, the relative error is a bit higher in this case. The relative error from the
nmag solution is as high as 0.028. The following plot shows the instability of nmag's energy
density for a simple 1D case with m = (cos(x), sin(x), 0).

.. image:: ../examples/exchange_demag/simple1D.png
    :scale: 75
    :align: center

For the demag energy density, our implementation follows the
nmag curve closer, and out relative error is approx. 0.006.

.. image:: ../examples/exchange_demag/demag_density.png
    :scale: 75
    :align: center

Time comparison between the nmag and finmag implementations shows the following:

.. include:: ../examples/exchange_demag/timings/results.rst
    :literal:

The complete code follows.

.. literalinclude:: ../examples/exchange_demag/test_exchange_demag.py

.. rubric:: Footnotes
