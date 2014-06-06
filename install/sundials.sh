#!/bin/bash

set -o errexit

ROOTDIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))



# The default installation location is $HOME. Set
# the SUNDIALS_PREFIX environment variable to change this.
SUNDIALS_PREFIX=${SUNDIALS_PREFIX:-$HOME}

read -p "Sundials will be installed in '$SUNDIALS_PREFIX' (this can be changed by setting the environment variable SUNDIALS_PREFIX). Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set SUNDIALS_PREFIX to the desired installation directory and try again."
    exit 0
fi

source=sundials-2.5.0

if ! [ -e ${SUNDIALS_PREFIX} ]; then
   install -d ${SUNDIALS_PREFIX};
   echo "Creating directory $SUNDIALS_PREFIX";
fi

cd $SUNDIALS_PREFIX
if [ ! -e $source.tar.gz ]
then
   wget http://ftp.mcs.anl.gov/pub/petsc/externalpackages/$source.tar.gz
fi

if [ ! -e $source ]
then
    tar xzvf $source.tar.gz
fi

cd $source
./configure --prefix=${SUNDIALS_PREFIX}/$source/sundials --enable-shared
#./configure
make
make install


echo "Please include ${SUNDIALS_PREFIX}/$source/sundials/include and add the path  ${SUNDIALS_PREFIX}/$source/sundials/lib to link the sundials."
