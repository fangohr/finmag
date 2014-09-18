#!/bin/bash

# downloads and installs oommf (and prerequisites)
# installs into user's home by default (can be changed with $OOMMF_PREFIX)
# places an oommf command in /usr/local/bin

set -o errexit

OOMMF_VERSION=oommf12a5bis_20120928
OOMMF_TARBALL=${OOMMF_VERSION}.tar.gz
OOMMF_URL=http://math.nist.gov/oommf/dist/${OOMMF_TARBALL}

TCLTKVERSION=${TCLTKVERSION:-8.6}

# use OOMMF_PREFIX to change installation location (default is $HOME)
OOMMF_PREFIX=${OOMMF_PREFIX:-$HOME}

# use APT_GET_INSTALL=-y to avoid apt-get asking for confirmation
APT_GET_OPTIONS=${APT_GET_OPTIONS:-}

# install prerequisites if needed
PKGS="tk$TCLTKVERSION-dev tcl$TCLTKVERSION-dev"
echo "Using TCLTKVERSION=${TCLTKVERSION}."
for pkg in $PKGS; do
    if ! dpkg -s $pkg > /dev/null 2>&1; then
	echo "OOMMF needs the package $pkg. Trying to install it..."
	sudo apt-get ${APT_GET_OPTIONS} install $pkg
    fi
done
read -p "OOMMF will be installed in $OOMMF_PREFIX (this can be changed by setting the environment variable OOMMF_PREFIX). Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set OOMMF_PREFIX to the desired installation directory and try again."
    exit 0
fi

# create installation directory if it doesn't exist
if ! [ -e ${OOMMF_PREFIX} ]; then
   echo "Creating directory $OOMMF_PREFIX.";
   install -d ${OOMMF_PREFIX};
fi

# download oommf
cd $OOMMF_PREFIX
if [ ! -e "$OOMMF_TARBALL" ]
then
    wget $OOMMF_URL
fi

# extract oommf
tar -xzf $OOMMF_TARBALL
OOMMF_EXTRACTED_DIR=$(echo $OOMMF_VERSION | sed -r 's/oommf([0-9])([^_]*).*/oommf-\1\.\2/')
mv $OOMMF_EXTRACTED_DIR oommf
cd oommf

# install oommf
ARCH=$(dpkg-architecture -qDEB_HOST_MULTIARCH)
export OOMMF_TCL_CONFIG=/usr/lib/${ARCH}/tcl${TCLTKVERSION}/tclConfig.sh
export OOMMF_TK_CONFIG=/usr/lib/${ARCH}/tk${TCLTKVERSION}/tkConfig.sh
tclsh$TCLTKVERSION oommf.tcl pimake distclean
tclsh$TCLTKVERSION oommf.tcl pimake upgrade
tclsh$TCLTKVERSION oommf.tcl pimake
tclsh$TCLTKVERSION oommf.tcl +platform

# create an executable called 'oommf' to call oommf in /usr/local/bin
oommf_command=$(cat <<EOF
#! /bin/bash
tclsh $OOMMF_PREFIX/oommf/oommf.tcl "\$@"
EOF
)

echo "Permissions needed to create an executable in '/usr/local/bin', and to\
 fix permissions in the $OOMMF_PREFIX/oommf directory."

sudo sh -c "echo '$oommf_command' > '/usr/local/bin/oommf'"
sudo chmod a+x /usr/local/bin/oommf

# Fix permissions (some read permissions seem to be missing by default)
chmod -R a+r $OOMMF_PREFIX/oommf
