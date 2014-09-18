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
# This section of the installation script is responsible for applying patches
# to the code.
#
# Firstly, the reversibility of the patches is determined; if the patches can
# be reversed then they have been applied already. In this case, no patching
# needs to be done.
#
# The 'sed' substitutions are necessary to change the installation directory to
# MAGPAR_PREFIX instead of HOME, but this has the unfortunate effect of making
# the first patch irreversible, since it cares about this line. Hence we try to
# reverse the 'sed' substitution first to counter this. These 'sed'
# substitutions are not patches themselves because patches would have no
# knowledge about MAGPAR_PREFIX.

# We attempt to reverse the 'sed' substitution. On the first execution of this
# script this will achieve nothing, but on subsequent runs it will reverse the
# following substitution. However, this will only work if MAGPAR_PREFIX is the
# same as in the first execution. Otherwise, Magpar will be installed untarred
# somewhere else anyway.
sed -i -e "s|MAGPAR_HOME = ${MAGPAR_PREFIX}/magpar-0.9|MAGPAR_HOME = \$(HOME)/magpar-0.9|" Makefile.in.defaults

# If the patch is not reversible, apply the patch. We enter this conditional
# block on the first execution, but not afterwards. This line might result in
# something like:
#     1 out of 1 hunk FAILED
#     2 out of 2 hunks FAILED
# being printed. This seems to be acceptable.
if ! patch -f -R -s -p1 --dry-run < ${ROOTDIR}/magpar.patch; then
    patch -p1 < ${ROOTDIR}/magpar.patch
    echo "Patch 'magpar.patch' SUCCESSFULLY applied."
else
    echo "Patch 'magpar.patch' already applied. Skipping..."
fi

# Likewise for the second patch.
if ! patch -f -R -s -p1 --dry-run < ${ROOTDIR}/magpar_code.patch; then
    patch -p1 < ${ROOTDIR}/magpar_code.patch
    echo "Patch 'magpar_code.patch' SUCCESSFULLY applied."
else
    echo "Patch 'magpar_code.patch' already applied. Skipping..."
fi

# Finally, we perform the sed substitution. This substitution will always be
# made.
sed -i -e "s|MAGPAR_HOME = \$(HOME)/magpar-0.9|MAGPAR_HOME = ${MAGPAR_PREFIX}/magpar-0.9|" Makefile.in.defaults

exit

make -f Makefile.libs
make

echo "Please add ${MAGPAR_PREFIX}/magpar-0.9/src/magpar.exe to your path."
