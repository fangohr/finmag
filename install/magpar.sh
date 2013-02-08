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
