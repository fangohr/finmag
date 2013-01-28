#!/bin/bash

# Check for required package
PKG=libfftw3-dev
if ! dpkg -s $PKG > /dev/null 2>&1; then
    echo "Need the package $PKG. Trying to install it..."
    sudo apt-get install $PKG
fi

# The default installation location is $HOME. Set
# the NFFT_PREFIX environment variable to change this.
NFFT_PREFIX=${NFFT_PREFIX:-$HOME}  # or maybe use NFFT_PREFIX=/usr/local ?

read -p "NFFT will be installed in '$NFFT_PREFIX' (this can be changed by setting the environment variable NFFT_PREFIX). Is this correct? (y/n)" -r
echo

if ! [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Please set NFFT_PREFIX to the desired installation directory and try again."
    exit 0
fi

# create installation directory if it doesn't exist
if ! [ -e ${NFFT_PREFIX} ]; then
   install -d ${NFFT_PREFIX};
   echo "Creating directory $NFFT_PREFIX";
fi

source=nfft-3.2.0

bak=`pwd`

cd $NFFT_PREFIX
if [ ! -e $source.tar.gz ]
then
    wget http://www-user.tu-chemnitz.de/~potts/nfft/download/$source.tar.gz
fi

if [ ! -e $source ]
then
    tar xzvf $source.tar.gz
fi



cd $source
./configure --prefix=${NFFT_PREFIX}
make
make install

echo "================================================================"
echo "To have access to libnfft, please add the following line to your"
echo "shell configuration file (for example .bashrc):"
echo ""
echo "   export LD_LIBRARY_PATH=\"${NFFT_PREFIX}/lib:\$LD_LIBRARY_PATH\""
echo "================================================================"

#echo "export LD_LIBRARY_PATH=${NFFT_PREFIX}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

cd $bak
cd ../sandbox/nfft/
NFFT_DIR=${NFFT_PREFIX} python setup.py build_ext --inplace
