#!/bin/bash

set -o errexit

# Check for required package 'gfortran'
if ! dpkg -s gfortran > /dev/null 2>&1; then
    echo "Magpar needs the package gfortran. Trying to install it..."
    sudo apt-get install gfortran
fi

# The default installation location is $HOME. Set
# the MAGPAR_PREFIX environment variable to change this.
MAGPAR_PREFIX=${MAGPAR_PREFIX:-$HOME}

echo "Installing magpar in '$MAGPAR_PREFIX'. Set the MAGPAR_PREFIX environment variable to specify a different location."

source=magpar-0.9

if ! [ -e ${MAGPAR_PREFIX} ]; then
   install -d ${MAGPAR_PREFIX};
   echo "Creating directory $MAGPAR_PREFIX";
fi
cp magpar.patch $MAGPAR_PREFIX
cp magpar_code.patch $MAGPAR_PREFIX
cd $MAGPAR_PREFIX
if [ ! -e $source.tar.gz ]
then
   wget http://www.magpar.net/static/$source/download/$source.tar.gz
fi

if [ ! -e $source ]
then
    tar xzvf $source.tar.gz
fi

mv magpar.patch $source/src
mv magpar_code.patch $source/src

cd $source/src
patch -p1 < magpar.patch
patch -p1 < magpar_code.patch

sed -i -e "s|MAGPAR_HOME = \$(HOME)/magpar-0.9|MAGPAR_HOME = ${MAGPAR_PREFIX}/magpar-0.9|" Makefile.in.defaults

make -f Makefile.libs
make

echo "Please add ${MAGPAR_PREFIX}/magpar-0.9/src/magpar.exe to your path."
