#!/bin/bash

set -o errexit

VAGRANT_HOME=$(echo ~vagrant)

# This script installs OOMMF, Magpar and Nmag system-wide. It also
# adds a couple of files in /etc/profile.d/ which adapt the PATH
# variable so that Magpar and Nmag can be found.


# Location where OOMMF, Magpar and Nmag are going to be installed
LOCALBASE=/usr/local/software
mkdir -p ${LOCALBASE}

# Install OOMMF
export TCLTKVERSION=8.5
export APT_GET_OPTIONS=-y
export OOMMF_PREFIX=${LOCALBASE}
cd ${VAGRANT_HOME}
yes | bash /vagrant/install_scripts/oommf.sh

# Install Magpar
# We create symolic link to the magpar executable in ${LOCALBASE}/bin
# so that we can add it to PATH without cluttering PATH with other stuff
# in the magpar directory.
export MAGPAR_PREFIX=${LOCALBASE}
echo "LOCALBASE: ${LOCALBASE}"
echo "MAGPAR_PREFIX: ${MAGPAR_PREFIX}"
cd ${VAGRANT_HOME}
yes | bash /vagrant/install_scripts/magpar.sh
mkdir -p ${LOCALBASE}/bin
ln -s ${LOCALBASE}/magpar-0.9/src/magpar.exe ${LOCALBASE}/bin/magpar.exe
echo "# Add Magpar to PATH" >> /etc/profile.d/magpar.sh
echo "export PATH=${LOCALBASE}/bin:\$PATH" >> /etc/profile.d/magpar.sh

# Install Nmag
export NMAG_PREFIX=${LOCALBASE}
cd ${VAGRANT_HOME}
yes | bash /vagrant/install_scripts/nmag.sh
echo "# Add Nmag to PATH" >> /etc/profile.d/nmag.sh
echo "export PATH=${LOCALBASE}/nmag-0.2.1/nsim/bin:\$PATH" >>  /etc/profile.d/nmag.sh
