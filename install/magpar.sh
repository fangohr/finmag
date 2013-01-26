#!/bin/bash

set -o errexit

# Check for required package 'gfortran'
if ! dpkg -s gfortran > /dev/null 2>&1; then
    echo "Magpar needs the package gfortran. Trying to install it..."
    sudo apt-get install gfortran
fi

# The default installation location is $HOME. Set
# the PREFIX environment variable to change this.
PREFIX=${PREFIX:-$HOME}

echo "Installing magpar in '$PREFIX'. Set the PREFIX environment variable to specify a different location."

source=magpar-0.9

if ! [ -e ${PREFIX} ]; then
   install -d ${PREFIX};
   echo "Creating directory $PREFIX";
fi
cp magpar.patch $PREFIX
cp magpar_code.patch $PREFIX
cd $PREFIX
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

sed -i -e "s|MAGPAR_HOME = \$(HOME)/magpar-0.9|MAGPAR_HOME = ${PREFIX}/magpar-0.9|" Makefile.in.defaults

make -f Makefile.libs
make

echo "Please add magpar-0.9/src/magpar.exe to your path."
