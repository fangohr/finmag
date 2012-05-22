Spatially varying anisotropy
============================

We demonstrate how an anisotropy can be used in simulations that depends on space. In this example, we think of a bilayer sample of size
:math:`L_x \times L_y \times L_z` where the uniaxial anisotropy easy axis is perpendicular to the film plane (and we think of the film as being in the x-y plane), i.e.

.. math::
   
   \vec{a}_1 = [0,0,1] \quad \mathrm{for} \quad \le L_z/2

and 

.. math:: 

   \vec{a}_1 = [1,0,0] \quad \mathrm{for} \quad z>L_z/2


The resulting magnetisation pattern reflects the anisotropy, and how -- due to the exchange -- the magnetisation changes slowly from perpendicular to in plane as we go from :math:`z=0` to :math`z=L_z`.

.. comment:

  .. image:: ../examples/spatially-varying-anisotropy/exchangespring.png
      :scale: 75
      :align: center
  
We plot particular components of the magnetisation and the anisotropy:

.. image:: ../examples/spatially-varying-anisotropy/profile.png
    :scale: 75
    :align: center

The code used here is

.. literalinclude:: ../examples/spatially-varying-anisotropy/run.py

