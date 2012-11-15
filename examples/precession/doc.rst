Switching off the precession term in the LLG
============================================

If we are only interested in the final (meta-stable) configuration of a run,
we can switch off the precession term in the Laundau Lifshitz Gilbert equation.

Simulation resolve the precession of the magnetisation around the effective
field by default, but this can be turned off by setting the attribute
do_precession of the LLG class to False, as the highlighted line in the
following code listing shows.

.. literalinclude:: ../examples/precession/run.py
    :language: python
    :emphasize-lines: 7
    :pyobject: run_simulation

While the time-development of the system without precession happens at the
same time scale as for the system with the precession term, the computation of
the system without the precession is significantly faster.

.. image:: ../examples/precession/precession.png 

Note, that the "*dynamics*" shown here are of course artificial and only used
to obtain a meta-stable physical configuration more efficiently.
