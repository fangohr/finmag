#!/bin/bash

#
# This script will download and install oommf into a directory
# called oommf under $OOMMF_PREFIX, which is your user's home per default. It will
# then create a command called "oommf" in your /usr/local/bin directory.
#
# It will also install the needed ubuntu packages.
#

set -o errexit

# Install prerequisites if needed
PKGS="tk-dev tcl-dev"
for pkg in $PKGS; do
    if ! dpkg -s $pkg > /dev/null 2>&1; then
	echo "OOMMF needs the package $pkg. Trying to install it..."
	sudo apt-get install $pkg
    fi
done

# The default installation location is $HOME. Set
# the OOMMF_PREFIX environment variable to change this.
OOMMF_PREFIX=${OOMMF_PREFIX:-$HOME}

read -p "OOMMF will be installed in '$OOMMF_PREFIX' (this can be changed by setting the environment variable OOMMF_PREFIX). Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set OOMMF_PREFIX to the desired installation directory and try again."
    exit 0
fi

# create installation directory if it doesn't exist
if ! [ -e ${OOMMF_PREFIX} ]; then
   install -d ${OOMMF_PREFIX};
   echo "Creating directory $OOMMF_PREFIX";
fi

# download and extract oommf
cd $OOMMF_PREFIX
if [ ! -e "oommf12a4pre-20100719bis.tar.gz" ]
then
    wget http://math.nist.gov/oommf/snapshot/oommf12a4pre-20100719bis.tar.gz
fi
tar -xzf oommf12a4pre-20100719bis.tar.gz
mv oommf12a4pre-20100719bis oommf
cd oommf

# install oommf
#OOMMF_TCL_INCLUDE_DIR=/usr/include/tcl8.5/; export OOMMF_TCL_INCLUDE_DIR
#OOMMF_TK_INCLUDE_DIR=/usr/include/tcl8.5/; export OOMMF_TK_INCLUDE_DIR
export OOMMF_TCL_CONFIG='/usr/lib/tcl8.6/tclConfig.sh'
export OOMMF_TK_CONFIG='/usr/lib/tk8.6/tkConfig.sh'
tclsh8.6 oommf.tcl pimake distclean
tclsh8.6 oommf.tcl pimake upgrade
tclsh8.6 oommf.tcl pimake
tclsh8.6 oommf.tcl +platform

# create an executable called 'oommf' to call oommf in /usr/local/bin
oommf_command=$(cat <<EOF
#! /bin/bash
tclsh $OOMMF_PREFIX/oommf/oommf.tcl "\$@"
EOF
)
sudo sh -c "echo '$oommf_command' > '/usr/local/bin/oommf'"
sudo chmod a+x /usr/local/bin/oommf

# Fix permissions (some read permissions seem to be missing by default)
chmod -R a+r $OOMMF_PREFIX/oommf
