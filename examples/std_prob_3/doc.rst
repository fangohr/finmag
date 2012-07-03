Standard Problem 3
==================

A cube with edge length :math:`L`, expressed in units of the intrinsic length scale, :math:`l_{ex}=\sqrt{A/K_m}\,m` is simulated, where :math:`A` is the exchange coupling strength and :math:`K_m=\frac{1}{2}\mu_0\MS^2\,kg\,m^{-2}\,s^{-2}` the magnetostatic energy density.
The uniaxial anisotropy paramater is :math:`K_u` with :math:`K_u = K_m/10`, and the easy axis is directed parallel to a principal axis of the cube.

Depending on the edge length :math:`L`, the total energy of the magnetisation will either be lower for a modified single domain state called flower state or a vortex state.
The problem now consists in finding the value of :math:`L` for which the two states have equal energy.
That value of :math:`L` is called the single domain limit.
The individual energy contributions and the spatially averaged magnetisation need to be recorded as well.

.. include:: ./doc_table.rst

The code used here is

.. literalinclude:: ../examples/std_prob_3/run.py
