#!/bin/bash

set -o errexit

# The user can say 'export APT_GET_INSTALL=-y' to avoid apt-get
# asking for confirmation.
APT_GET_OPTIONS=${APT_GET_OPTIONS:-}

# Check for required packages
PKGS="g++ libblas-dev libreadline-dev make m4 gawk zlib1g-dev readline-common liblapack-dev"
for pkg in $PKGS; do
    if ! dpkg -s $pkg > /dev/null 2>&1; then
	echo "Nmag needs the package $pkg. Trying to install it..."
	sudo apt-get ${APT_GET_OPTIONS} install $pkg
    fi
done

# The default installation location is $HOME. Set
# the NMAG_PREFIX environment variable to change this.
NMAG_PREFIX=${NMAG_PREFIX:-$HOME}  # or maybe use NMAG_PREFIX=/usr/local ?

read -p "Nmag will be installed in '$NMAG_PREFIX' (this can be changed by setting the environment variable NMAG_PREFIX). Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set NMAG_PREFIX to the desired installation directory and try again."
    exit 0
fi

# create installation directory if it doesn't exist
if ! [ -e ${NMAG_PREFIX} ]; then
   install -d ${NMAG_PREFIX};
   echo "Creating directory $NMAG_PREFIX";
fi

source="nmag-0.2.1"
TARBALLNAME="$source.tar.gz"
TARBALLURL="http://nmag.soton.ac.uk/nmag/0.2/download/all/$TARBALLNAME"

# Nmag patch location
NMAGPATCHPATH=`pwd`/nmag-patch-cpp-comment-2014-05-12.patch

echo "Installing $source from $TARBALLURL"

echo "Changing directory to $NMAG_PREFIX"
cd $NMAG_PREFIX
echo "Working directory is `pwd`"
pwd

if [ ! -e $TARBALLNAME ]
then
    wget $TARBALLURL
fi

if [ ! -e $source ]
then
    tar xzvf $TARBALLNAME

fi

cd $source

# With more recent version of gcc (for examlpe gcc4.8), an error is
# raised when compiling Nmag's version of hdf5. It is just a C++ comment
# in a C file. (in nmag-0.2.1/hdf5/tools/lib/h5diff.c)
# This patch converts it to a C comment.

make .deps_hdf5_untar  # This should make the path available for the patch
                       # (though this feels dirty)

pushd hdf5/tools/lib

# If the patch is not reversible, apply the patch. We enter this conditional
# block on the first execution, but not afterwards. This change was implemented
# to allow the installation script to continue execution if the patch has been
# applied previously. This block might result in something like:
#     1 out of 1 hunk FAILED
# being printed. This seems to be acceptable.
if ! patch -f -R -s --dry-run < $NMAGPATCHPATH; then
    patch < $NMAGPATCHPATH
    echo "Patch '$NMAGPATCHPATH' SUCCESSFULLY applied."
else
    echo "Patch '$NMAGPATCHPATH' already applied. Skipping..."
fi

popd
make

# Hack-ish fix because some executables in nmag are not installed with
# the correct permissions when installed into a non-standard location
# (probably only occurs when using sudo).  -- Max, 8.2.2013
cd $NMAG_PREFIX/nmag-0.2.1/nsim/bin  && \
    chmod 755 ncol nmagpp nmagpp_pre0_1_5000 nmagprobe nmeshimport nmeshmirror \
              nmeshpp nmeshsort nsim nsimexec nsim_i nsim-raw nsimversion
