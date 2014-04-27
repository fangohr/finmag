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
    python-setuptools python-progressbar paraview-python cython netgen netgen-doc"

building_doc="texlive-latex-extra texlive-latex-recommended texlive-fonts-recommended python-pygments"

suggested="mercurial ipython ipython-notebook grace gnuplot gmsh"

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

