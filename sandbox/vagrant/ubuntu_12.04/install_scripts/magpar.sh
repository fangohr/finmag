#!/bin/bash

set -o errexit

ROOTDIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))

# The user can say 'export APT_GET_INSTALL=-y' to avoid apt-get
# asking for confirmation.
APT_GET_OPTIONS=${APT_GET_OPTIONS:-}

# Check for required package 'gfortran'
if ! dpkg -s gfortran > /dev/null 2>&1; then
    echo "Magpar needs the package gfortran. Trying to install it..."
    sudo apt-get ${APT_GET_OPTIONS} install gfortran
fi

# The default installation location is $HOME. Set
# the MAGPAR_PREFIX environment variable to change this.
MAGPAR_PREFIX=${MAGPAR_PREFIX:-$HOME}

read -p "Magpar will be installed in '$MAGPAR_PREFIX' (this can be changed by setting the environment variable MAGPAR_PREFIX). Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set MAGPAR_PREFIX to the desired installation directory and try again."
    exit 0
fi

source=magpar-0.9

if ! [ -e ${MAGPAR_PREFIX} ]; then
   install -d ${MAGPAR_PREFIX};
   echo "Creating directory $MAGPAR_PREFIX";
fi
cd $MAGPAR_PREFIX
if [ ! -e $source.tar.gz ]
then
   wget http://www.magpar.net/static/$source/download/$source.tar.gz
fi

if [ ! -e $source ]
then
    tar xzvf $source.tar.gz
fi

cd $source/src
patch -N -p1 < ${ROOTDIR}/magpar.patch
patch -N -p1 < ${ROOTDIR}/magpar_code.patch

sed -i -e "s|MAGPAR_HOME = \$(HOME)/magpar-0.9|MAGPAR_HOME = ${MAGPAR_PREFIX}/magpar-0.9|" Makefile.in.defaults

make -f Makefile.libs
make

echo "Please add ${MAGPAR_PREFIX}/magpar-0.9/src/magpar.exe to your path."
