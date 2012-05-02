#!/bin/bash

#Required libraries
#
echo "If required, install required libraries using"
echo "apt-get install g++ libblas-dev libreadline-dev make m4 gawk zlib1g-dev readline-common liblapack-dev"

sudo apt-get install g++ libblas-dev libreadline-dev make m4 gawk zlib1g-dev readline-common liblapack-dev 


PREFIX="$HOME" # EDIT HERE.
#or maybe
#PREFIX=/usr/local/

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


