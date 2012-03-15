Time dependent external fields
==============================

In this example, we apply an oscillating external field, compute the time development of the magnetic system (no demag, no anisotropy, exchange irrelevant), plot the magnetisation as a function of time, and fit a sinusoidal function through one component (if scipy is installed).

Currently, the LLG object does not provide a direct way of allowing a
time dependent external applied field (we use either the term
'applied' or 'external' to mean the same thing). Instead, we manually override parts of the LLG class in the code below (see source code).


.. literalinclude:: ../examples/time-dependent-applied-field/test_appfield.py

The plot of the normalised magnetisation components (results.png)

.. image:: ../examples/time-dependent-applied-field/results.png
    :scale: 75
    :align: center


The fitted parameters are (fittedresults.txt):

.. literalinclude:: ../examples/time-dependent-applied-field/fittedresults.txt


.. image:: ../examples/time-dependent-applied-field/fit.png
    :scale: 75
    :align: center


.. rubric:: Footnotes

