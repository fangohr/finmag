#!/bin/bash

# Cause script to exit if command or pipe returns non-zero exit state.
set -e

# Help string.
if [ "$1" == "--help" -o "$1" == "-h" ]; then
    echo "This script contains some commands used by docker. It doesn't yet \
have a purpose, but will probably install docker, download the fenics docker \
container image, and install some things on it."
    exit
fi

# Change to local directory of this script.
cd $( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Panic if not on linux-gnu.
if [ $OSTYPE != "linux-gnu" ]; then
    echo "ERROR: This install script requires a linux-gnu operating system to \
run safely. Detected value for \$OSTYPE is ${OSTYPE}. Exiting." >&2
    exit
fi

# Install docker and required packages.
HOST_PACKAGES=docker.io
echo "Super user authentication required to download some packages: \
${HOST_PACKAGES}."
sudo apt-get update
sudo apt-get install -y $HOST_PACKAGES
sudo ln -sf /usr/bin/docker.io /usr/local/bin/docker
sudo sed -i '$acomplete -F _docker docker' /etc/bash_completion.d/docker.io

# Grab the fenics docker image.
FENICS_IMG=fenicsproject/stable-ppa
echo "Downloading the fenics docker image '${FENICS_IMG}' from the docker \
cloud."
sudo docker pull $FENICS_IMG
echo "Image '${FENICS_IMG}' downloaded successfully."

# Leave if the finmag binary is not available locally.
if ! [ -e finmag_ubuntu_14_04.deb ]; then
    echo "ERROR: Finmag debian package 'finmag_ubuntu_14_04.deb' is missing, \
place in the directory '${PWD}' as this script to proceed." >&2
    exit
fi

# "Provision" the docker image. --volume is used to mount a directory
# on the host machine in the container. See section 6.3.3 of the below URL for
# syntax.
# https://access.redhat.com/documentation/en-US/Red_Hat_Enterprise_Linux/7/html
# /Resource_Management_and_Linux_Containers_Guide/sec-Sharing_Data_Across_Conta
# iners.html
sudo docker run --volume=${PWD}:/home/:ro $FENICS_IMG bash -x /home/internal.sh
CONTAINER_ID="$(sudo docker ps -lq)"
IMAGE_NAME=finmag_fenics
sudo docker commit ${CONTAINER_ID} ${IMAGE_NAME}

# Import finmag.
sudo docker run $IMAGE_NAME /bin/bash -c \
"python -c 'import finmag; finmag.example.barmini().relax()'"
