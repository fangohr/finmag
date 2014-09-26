#!/bin/bash

set -o errexit

# Specify the Sundials version
SUNDIALS=sundials-2.5.0

# Read SUNDIALS_PREFIX from the environment. If it is not defined, use ../external_deps as default value.
SUNDIALS_PREFIX=$(readlink -f ${SUNDIALS_PREFIX:-../external_deps})

INSTALL_DIR=${SUNDIALS_PREFIX}/${SUNDIALS}
TMP_BUILD_DIR=$(mktemp -d)

# Ask for confirmation on install location
read -p "Sundials will be installed in the subdirectory '${SUNDIALS}' of the directory '${SUNDIALS_PREFIX}' (this can be changed by setting the environment variable SUNDIALS_PREFIX). Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set SUNDIALS_PREFIX to the desired installation directory and try again."
    exit 0
fi

if ! [ -e ${SUNDIALS_PREFIX} ]; then
   install -d ${SUNDIALS_PREFIX};
   echo "Creating directory $SUNDIALS_PREFIX";
fi

# Download and extract Sundials tarball in temporary build directory
echo "Building Sundials in the temporary directory '${TMP_BUILD_DIR}' (which will be deleted after successful installation)."
cd ${TMP_BUILD_DIR}
wget http://ftp.mcs.anl.gov/pub/petsc/externalpackages/$SUNDIALS.tar.gz
tar xzvf $SUNDIALS.tar.gz

# Build Sundials in temporary build directory and install it in $INSTALL_DIR
cd ${TMP_BUILD_DIR}/$SUNDIALS
./configure --prefix=${INSTALL_DIR} --with-cflags=-fPIC --enable-shared --enable-lapack --enable-mpi
make
make install

# Remove temporary build directory
echo "Removing directory ${TMP_BUILD_DIR}"
rm -rf ${TMP_BUILD_DIR}


echo "In order to use this installation, add '${SUNDIALS_PREFIX}/$SUNDIALS/include' to the include directories and point the linker to '${SUNDIALS_PREFIX}/$SUNDIALS/sundials/lib'."
