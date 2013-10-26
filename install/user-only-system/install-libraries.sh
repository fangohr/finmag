#!/bin/bash 
set -o errexit

if [ "$1" == "--help" -o "$1" == "-h" ]
then
    echo "
This script will attempt to install required libraries to run 
the finmag user code."

    exit
fi

required="fenics libboost-python-dev libboost-thread-dev libsundials-serial-dev
    libboost-test-dev python-matplotlib python-visual python-scipy python-pip
    python-setuptools python-progressbar"



required="fenics python-matplotlib python-visual python-scipy python-netifaces python-pip python-setuptools libsundials-serial ipython ipython-notebook ipython-qtconsole python-pygments python-py python-m2crypto grace gnuplot netgen netgen-doc gmsh mercurial " 

#libboost-python-dev libboost-thread-dev l
#    python-setuptools python-progressbar"

#building_doc="texlive-latex-extra texlive-latex-recommended python-pygments
#    texlive-fonts-recommended"

#suggested="mercurial ipython grace gnuplot netgen netgen-doc gmsh"

packages="$required"
if [ "$1" == "--all" ]
then
    packages="$packages $building_doc $suggested"
fi

sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install $packages
#sudo pip install -U sphinx pytest
