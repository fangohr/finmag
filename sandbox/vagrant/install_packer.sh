#!/bin/bash

set -e  # exit if an error occurs

PACKER_INSTALL_DIR=$(readlink -f ${PACKER_INSTALL_DIR:-./packer})
PACKER_VERSION=${PACKER_VERSION:-0.5.2_linux_amd64}

echo "Packer will be installed in the directory '${PACKER_INSTALL_DIR}'."
echo "This can be changed by setting the environment variable 'PACKER_INSTALL_DIR' to another value."
echo
read -p "Do you want to continue? (yes/no)" -r
#echo

if [[ "$REPLY" = "yes" || "$REPLY" = "Yes" || "$REPLY" = "YES" ]]
then
    if [ -e ${PACKER_INSTALL_DIR} ]; then
        if [ "$(ls -A ${PACKER_INSTALL_DIR})" ]; then
            echo "The directory '${PACKER_INSTALL_DIR}' exists and is not empty. Aborting."
            exit 1
        fi
    fi

    # Create the directory where packer will be installed (if it doesn't exist)
    echo "Creating directory ${PACKER_INSTALL_DIR}"
    mkdir -p ${PACKER_INSTALL_DIR}

    # Download the packer zip into that directory
    wget https://dl.bintray.com/mitchellh/packer/${PACKER_VERSION}.zip -O ${PACKER_INSTALL_DIR}/packer-${PACKER_VERSION}.zip

    # Extract the zip file
    unzip -d ${PACKER_INSTALL_DIR} ${PACKER_INSTALL_DIR}/packer-${PACKER_VERSION}.zip

    echo
    echo "Successfully installed packer in the directory '${PACKER_INSTALL_DIR}'."
    echo "To use it, add this directory to your PATH variable by executing"
    echo "the following command:"
    echo
    echo "    export PATH=\${PATH}:${PACKER_INSTALL_DIR}"
    echo
    echo "You may also want to add this line to your ~/.bashrc file to"
    echo "make it permanent."
    echo
    echo "If you ever want to uninstall packer, simply delete the directory "
    echo "where it was installed:"
    echo
    echo "    rm -rf ${PACKER_INSTALL_DIR}"
fi
