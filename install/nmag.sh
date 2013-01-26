#!/bin/bash

set -o errexit

# Check for required packages
PKGS="g++ libblas-dev libreadline-dev make m4 gawk zlib1g-dev readline-common liblapack-dev"
for pkg in $PKGS; do
    if ! dpkg -s $pkg > /dev/null 2>&1; then
	echo "Nmag needs the package $pkg. Trying to install it..."
	sudo apt-get install $pkg
    fi
done

# The default installation location is $HOME. Set
# the PREFIX environment variable to change this.
PREFIX=${PREFIX:-$HOME}  # or maybe use PREFIX=/usr/local ?

echo "Installing nmag in '$PREFIX'. Set the PREFIX environment variable to specify a different location."

# create installation directory if it doesn't exist
if ! [ -e ${PREFIX} ]; then
   install -d ${PREFIX};
   echo "Creating directory $PREFIX";
fi

source="nmag-0.2.1"
TARBALLNAME="$source.tar.gz"
TARBALLURL="http://nmag.soton.ac.uk/nmag/0.2/download/all/$TARBALLNAME"

echo "Installing $source from $TARBALLURL"

echo "Changing directory to $PREFIX"
cd $PREFIX
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

make
