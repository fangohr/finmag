LLG equation
============

The dynamics of the magnetisation field :math:`\vec{M}(\vec{r},t)` is governed by the Landau-Lifschitz and Gilbert (LLG) equation

.. math::

   \frac{\mathrm{d} \vec{M}}{\mathrm{d} t} =
     -\gamma_L \mu_0 \, \vec{M} \times \vec{H}
    - \frac{\alpha_L}{M_{\mathrm{sat}}} \, \vec{M} \times [ \vec{M} \times \vec{H}]

where :math:`\mu0 = 4\pi10^{−7}\mathrm{N/A^2}` denotes the permeability of free space (also known as the magnetic constant), :math:`\alpha_L` the Landau-Lifshitz damping parameter, and 

.. math::

  \gamma_L = \frac{\gamma_G}{1+\alpha_G^2}

and 

.. math::

  \alpha_L = \frac{\alpha_G\gamma_G}{1+\alpha_G^2}

and thus :math:`\alpha_L = \alpha_G\gamma_G` ([#Zimmermann2007]_ equation (2.21). 

It is common to use it :math:`\gamma_G = 2.210173e5 \mathrm{m/(As)}` ([#Scholz2003]_ after equation (3.7), [#OOMMFManual]_) which is also known  as the gyromagnetic ratio.

When we discuss a damping value :math:`\alpha` we normally refer to :math:`\alpha_L`. 

In the absence of all effective fields (such as demagnetisation, anisotropy and exchange fields) the magnetisation behaves like a macrospin, i.e. the magnetisation field :math:`\vec{M}(\vec{r},t)` is uniform and thus does not depend on the position: :math:`\vec{M}(t)`. By applying an external field, we can study the motion of this macrospin a static field, and this can be compared with an analytical solution for the system ([#Franchin2009]_, Appendix B, should add equation number here XXX).

[To add: setup of simulation, briefly, some results, including plots for different damping values.]




.. image:: ../examples/macrospin/alpha-1.00.png
    :scale: 75
    :align: center

Testing, done.


.. rubric:: Footnotes

.. [#Zimmermann2007] Jurgen Zimmermann, *Micromagnetic simulations of magnetic exchange spring systems*, PhD Thesis, University of Southampton, UK (2007), `pdf <http://eprints.soton.ac.uk/65551/>`__

.. [#Scholz2003] Werner Scholz, *Scalable Parallel Micromagnetic Solvers for Magnetic Nanostructures*, PhD Thesis, University of Vienna, Austria (2003), `pdf <http://www.cwscholz.net/projects/diss/diss.pdf>`__ 

.. [#OOMMFManual] OOMMF Manual (2003) `pdf <http://math.nist.gov/oommf/doc/userguide12a3/userguide12a3_20021030.pdf>`__

.. [#Franchin2009] Matteo Franchin, *Multiphysics simulations of magnetic nanostructures*, PhD Thesis, University of Southampton, UK (2009), `pdf <http://eprints.soton.ac.uk/161207/>`__

