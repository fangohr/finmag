# Need to run this with bash, not sh. 

set -o errexit

if [ "$1" == "--help" -o "$1" == "-h" ]
then
    echo "
This script will install required libraries to run finmag.

Packages to build the documentation and additional suggested
packages can be installed by appending an --all argument when
calling this script."
    exit
fi

required="fenics libboost-python-dev libboost-thread-dev libsundials-serial-dev
    libboost-test-dev python-matplotlib python-visual python-scipy python-pip
    python-setuptools python-progressbar paraview-python cython netgen netgen-doc python-zmq python-tornado"

#Python-zmq and python-tornado are requirements for the ipython notebook.

building_doc="texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended python-pygments"

suggested="mercurial grace gnuplot gmsh"

packages="$required"
if [ "$1" == "--all" ]
then
    packages="$packages $building_doc $suggested"
fi

sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install $packages
sudo pip install -U sphinx pytest aeon sh diff-match-patch

# [Observation: We need IPython installed to be able to import finmag. So
# we need to make sure to also install ipython.]

# the debian 1404 package for ipython is 1.2, i.e. too old, so install 
# via pip:
sudo pip install -U ipython
sudo pip install -U pyzmq

# The next step is important but not working yet (HF 12 May 2014)

# Eigenmodes need petsc4py
sudo apt-get install python-petsc4py


# the following seems to have worked on osiris with ubunt14.04 but not on 
# Hans virtual machine with Ubuntu 14.04.
export PETSC_DIR=/usr/lib/petsc
export SLEPC_DIR=/usr/lib/slepc
PETSC_DIR=/usr/lib/petsc SLEPC_DIR=/usr/lib/slepc sudo pip install https://bitbucket.org/slepc/slepc4py/downloads/slepc4py-3.4.tar.gz


