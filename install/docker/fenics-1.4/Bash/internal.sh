#!/bin/bash

cd /home/

# Install dependencies for the finmag binary.
CONTAINER_PACKAGES="python-m2crypto python-netifaces git gmsh fenics \
libboost-python-dev libboost-thread-dev libsundials-serial-dev \
libboost-test-dev python-matplotlib python-visual python-scipy python-pip \
python-setuptools python-progressbar paraview-python cython netgen netgen-doc \
python-zmq python-tornado"

CONTAINER_PIP="sphinx pytest aeon sh diff-match-patch"
apt-get install -y software-properties-common
add-apt-repository ppa:fenics-packages/fenics
apt-get update
apt-get dist-upgrade
apt-get install -y ${CONTAINER_PACKAGES}
pip install -U ${CONTAINER_PIP}

# Install the finmag binary that is here.
dpkg -i /home/finmag_ubuntu_14_04.deb
