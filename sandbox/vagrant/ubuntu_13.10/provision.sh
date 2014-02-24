#!/bin/bash

set -o errexit

echo "I am working as $(whoami)"
echo "hostname: $(hostname)"

VAGRANT_HOME=$(echo ~vagrant)

# Install FEniCS (development version)
apt-get update
apt-get -y install python-software-properties  # to make the command 'add-apt-repository' available
add-apt-repository ppa:fenics-packages/fenics
add-apt-repository ppa:fenics-packages/fenics-dev
apt-get update
apt-get -y install fenics libdolfin1.3-dev

# Install Paraview 3.98 through a custom PPA (provided at https://launchpad.net/~gladky-anton/+archive/paraview)
add-apt-repository ppa:gladky-anton/paraview
apt-get update
apt-get -y install paraview

# Install required packages for compilation from source
apt-get install -y mercurial python-setuptools xpra libsundials-serial-dev libboost-python-dev libboost-test-dev netgen gmsh
#exit
easy_install -U pip
pip install -U distribute
pip install -U cython aeon sh matplotlib ipython pytest progressbar numpy scipy

# Tidy up packages that are no longer needed because newer versions have been installed
apt-get -y autoremove

# Add Bitbucket to the list of known hosts
sudo -u vagrant ssh-keyscan -H bitbucket.org >> ${VAGRANT_HOME}/.ssh/known_hosts

# Copy private and public key from Jenkins so that we can check out
# the Finmag repository without providing a password.
cp /vagrant/shared_folder/jenkins_id_rsa ${VAGRANT_HOME}/.ssh/id_rsa
cp /vagrant/shared_folder/jenkins_id_rsa.pub ${VAGRANT_HOME}/.ssh/id_rsa.pub
chmod 400 ${VAGRANT_HOME}/.ssh/id_rsa
chown vagrant:vagrant ${VAGRANT_HOME}/.ssh/id_rsa
chown vagrant:vagrant ${VAGRANT_HOME}/.ssh/id_rsa.pub

# Clone Finmag repository and add it to the PYTHONPATH
if ! [ -e finmag ]; then
    sudo -u vagrant hg clone ssh://hg@bitbucket.org/fangohr/finmag
fi
echo "" >> ${VAGRANT_HOME}/.bashrc
echo "# Add Finmag repository to PYTHONPATH" >> ${VAGRANT_HOME}/.bashrc
echo "export PYTHONPATH=~/finmag/src:\${PYTHONPATH}" >> ${VAGRANT_HOME}/.bashrc

# Location where OOMMF, Magpar and Nmag are going to be installed
LOCALBASE=/usr/local/software
mkdir -p ${LOCALBASE}

# Install OOMMF
export TCLTKVERSION=8.5
export APT_GET_OPTIONS=-y
export OOMMF_PREFIX=${LOCALBASE}
cd ${VAGRANT_HOME}
yes | bash ./finmag/install/oommf.sh

# Install Magpar
# We create symolic link to the magpar executable in ${LOCALBASE}/bin
# so that we can add it to PATH without cluttering PATH with other stuff
# in the magpar directory.
export MAGPAR_PREFIX=${LOCALBASE}
cd ${VAGRANT_HOME}
yes | bash ./finmag/install/magpar.sh
mkdir -p ${LOCALBASE}/bin
ln -s ${LOCALBASE}/magpar-0.9/src/magpar.exe ${LOCALBASE}/bin/magpar.exe
echo "export PATH=${LOCALBASE}/bin:\$PATH" >> ${VAGRANT_HOME}/.bashrc

# Install Nmag
export NMAG_PREFIX=${LOCALBASE}
cd ${VAGRANT_HOME}
yes | bash ./finmag/install/nmag.sh
echo "" >> ${VAGRANT_HOME}/.bashrc
echo "# Add Nmag to PATH" >> ${VAGRANT_HOME}/.bashrc
echo "export PATH=${LOCALBASE}/nmag-0.2.1/nsim/bin:\$PATH" >> ${VAGRANT_HOME}/.bashrc

# Run Finmag test suite (we need xpra to provide an X display
# because the virtual machine is running in a headless mode).
xpra start :1
export DISPLAY=:1
export TEST_OPTIONS=-svx
sudo -u vagrant make -C ${VAGRANT_HOME}/finmag test
#sudo -u vagrant make pytest-fast
