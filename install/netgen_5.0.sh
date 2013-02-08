#!/bin/bash

set -o errexit

# Install prerequisites if needed
PKGS="libtogl-dev libxmu-dev stow"
for pkg in $PKGS; do
    if ! dpkg -s $pkg > /dev/null 2>&1; then
        echo "Netgen (or this script) needs the package $pkg. Trying to install it..."
        sudo apt-get install $pkg
    fi
done

NETGEN_PREFIX=/usr/local/stow/
NETGEN_BUILDDIR=${NETGEN_PREFIX}/netgen_build

echo "The default install location for Netgen is /usr/local/stow/."
echo "This makes it possible to use the 'stow' utility to make it "
echo "appear as if Netgen was installed in /usr/local/ directly "
echo "(which is what this script will do), but also allows a clean "
echo "uninstall via 'stow -D'."
echo
echo "Netgen will be installed in '$NETGEN_PREFIX'"
echo "(this can be changed by setting the environment variable NETGEN_PREFIX)."
read -p "Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set NETGEN_PREFIX to the desired installation directory and try again."
    exit 0
fi

# create installation directory if it doesn't exist
if ! [ -e ${NETGEN_PREFIX} ]; then
   install -d ${NETGEN_PREFIX};
   echo "Creating directory $NETGEN_PREFIX";
fi

# Download and unpack
install -d $NETGEN_BUILDDIR
cd $NETGEN_BUILDDIR
wget http://downloads.sourceforge.net/project/netgen-mesher/netgen-mesher/5.0/netgen-5.0.0.tar.gz
tar xzvf netgen-5.0.0.tar.gz
cd ./netgen-5.0.0

# Configure
/bin/sh ./configure --prefix=${NETGEN_PREFIX}/netgen-5.0.0

# Build
make

# Install into NETGEN_PREFIX
make install

# Use stow to make it appear as if netgen is installed
# in /usr/local/ directly.
#
# XXX TODO: We should check that NETGEN_PREFIX doesn't
# point to anything else. Maybe the user doesn't want
# to use stow?
cd $NETGEN_PREFIX && stow -v netgen-5.0.0

echo
echo "Installation is complete. Please set the environment"
echo "variable NETGENDIR to the directory containing ng.tcl"
echo "This is probably ${NETGEN_PREFIX}/netgen-5.0.0/bin."
echo "It might be a good idea to set this system-wide, for"
echo "example in the file /etc/profile.d/netgen.sh"
