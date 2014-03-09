#!/bin/bash

set -o errexit

echo "I am working as $(whoami)"
echo "hostname: $(hostname)"

VAGRANT_HOME=$(echo ~vagrant)

# Install Finmag prerequisites
bash /vagrant/install_scripts/finmag_prerequisites.sh

# Install OOMMF, Magpar and Nmag
bash ./provision__oommf_magpar_nmag.sh

# Clone finmag and finmag-dist into vagrant home directory
bash ./provision__clone_finmag_and_finmag-dist.sh

# Build binary tarball and .deb package
cd ${VAGRANT_HOME}/finmag-dist
sudo -u vagrant python dist-wrapper.py --finmag-repo=${VAGRANT_HOME}/finmag --build-deb=ubuntu_12_04

# Install newly built .deb package system-wide
dpkg -i ${VAGRANT_HOME}/finmag-dist/finmag_ubuntu_12_04*.deb

# # Run Finmag test suite. Note that we need xpra to provide
# # an X display because the virtual machine is running in
# #  headless mode).
# #xauth add :0 . $(mcookie)
# xpra start :1
# export DISPLAY=:1
# export TEST_OPTIONS=-svx
# sudo -u vagrant make -C ${VAGRANT_HOME}/finmag test
# #sudo -u vagrant make pytest-fast

# TODO:
#
# - [ ] Copy documentation IPython notebooks into /home/vagrant/finmag_docs
# - [ ] Create new user 'finmag' (with password 'finmag'); make sure that a basic .bashrc file is created
# - [ ] Run python -c "import finmag" (to create the .finmagrc file)
# - [ ] Adapt .finmagrc so that xpra=False by default
# - [ ] Add line to ~/.bash_aliases (until we migrate to a newer version of Paraview):
#          alias paraview='paraview --use-old-panels'
# - [ ] Remove ~/finmag and ~/finmag-dist.
