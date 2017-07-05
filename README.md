[![CircleCI](https://circleci.com/gh/fangohr/finmag.svg?style=svg&circle-token=6e89ca6e2d8bb3dadd4ac9ec84bec71d91336f9c)](https://circleci.com/gh/fangohr/finmag)
<img src="logos/finmag_logo.png" width="300" align="right">

FinMag
======

- a thin (and mostly) Python layer (hopefully!) on top of
FEniCS/Dolfin to enable Python-scripted multi-physics micromagnetic
simulations.

- finmag solves micromagnetic problems using finite elements

- executes not efficiently in parallel (time integration is serial)

- the name FINmag originates from the dolFIN interface to FEniCS

- The GitHub page of the project is https://github.com/fangohr/finmag

- The code is developed by Hans Fangohr's group with contributions from
  Anders Johansen, Dmitri Chernyshenko, Gabriel Balaban, Marc-Antonio
  Bisotti, Maximilian Albert, Weiwei Wang, Marijan Beg, Mark Vousden,
  Beckie Carey, Ryan Pepper, Leoni Breth, and Thomas Kluyver at the
  University of Southampton.


- This is a working prototype - not polished, with some (in large parts
  outdated) attempts of documentation. Contributions and pull requests
  to both the code and documentation are welcome.

No support is available.


Documentation
-------------
The documentation is available in the form of Jupyter notebooks is
available in `doc/ipython_notebooks_src` directory.


Installation
------------
Finmag dependencies can be installed by running an appropriate script
(for the Ubuntu version you use) in `install/` directory or by making
a Docker container with Dockerfile in `install/docker/`.


If you decide to install dependencies using a shell script in
`install/` directory, you will need to add the path to the
`finmag/src/` directory to your $PYTHONPATH.


# How to cite Finmag

Finmag, University of Southampton, Hans Fangohr and team (2017)

# Acknowledgement

We acknowledge financial support from

- EPSRC’s Doctoral Training Centre in Complex System Simulation
  (EP/G03690X/1), http://icss.soton.ac.uk

- EPSRC's Centre for Doctoral Training in Next Generation
Computational Modelling (#EP/L015382/1), http://ngcm.soton.ac.uk

- Horizon 2020 European Research Infrastructure project OpenDreamKit
  (Project ID 676541).

- UK EPSRC Programme grant Skyrmionics (EP/N032128/1)
