#!/bin/bash

set -o errexit

# This script clones the source repositories 'finmag' and
# 'finmag-dist' from Bitbucket and places them in the home
# directory of the user 'vagrant'.

VAGRANT_HOME=$(echo ~vagrant)

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
cd ${VAGRANT_HOME}
if ! [ -e finmag ]; then
    sudo -u vagrant hg clone ssh://hg@bitbucket.org/fangohr/finmag
fi
echo "" >> ${VAGRANT_HOME}/.bashrc
echo "# Add Finmag repository to PYTHONPATH" >> ${VAGRANT_HOME}/.bashrc
echo "export PYTHONPATH=~/finmag/src:\${PYTHONPATH}" >> ${VAGRANT_HOME}/.bashrc

# Clone finmag-dist repository
cd ${VAGRANT_HOME}
if ! [ -e finmag-dist ]; then
    sudo -u vagrant hg clone ssh://hg@bitbucket.org/fangohr/finmag-dist
fi
