#!/bin/bash

set -e

# Install FEniCS (development version)
apt-get update
apt-get -y install python-software-properties  # to make the command 'add-apt-repository' available
add-apt-repository ppa:fenics-packages/fenics
#add-apt-repository ppa:fenics-packages/fenics-dev
apt-get update
apt-get -y install fenics libdolfin1.3-dev

# Install Paraview 3.98 through a custom PPA (provided at https://launchpad.net/~gladky-anton/+archive/paraview)
add-apt-repository ppa:gladky-anton/paraview
apt-get update
apt-get -y install paraview

# Install required packages for compilation from source
apt-get install -y mercurial python-setuptools libsundials-serial-dev libboost-python-dev libboost-test-dev netgen gmsh python-m2crypto python-netifaces xserver-xorg xpra vim emacs23 git gitk
easy_install -U pip
pip install -U distribute
pip install -U cython aeon sh matplotlib ipython pytest progressbar numpy scipy sumatra parameters

# Tidy up packages that are no longer needed because newer versions have been installed
apt-get -y autoremove
