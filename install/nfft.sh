#!/bin/bash

# Check for required package
PKG=libfftw3-dev
if ! dpkg -s $PKG > /dev/null 2>&1; then
    echo "Need the package $PKG. Trying to install it..."
    sudo apt-get install $PKG
fi

# The default installation location is $HOME. Set
# the PREFIX environment variable to change this.
PREFIX=${PREFIX:-$HOME}  # or maybe use PREFIX=/usr/local ?

echo "Installing nfft in '$PREFIX'. Set the PREFIX environment variable to specify a different location."

# create installation directory if it doesn't exist
if ! [ -e ${PREFIX} ]; then
   install -d ${PREFIX};
   echo "Creating directory $PREFIX";
fi

source=nfft-3.2.0

bak=`pwd`

cd $PREFIX
if [ ! -e $source.tar.gz ]
then
    wget http://www-user.tu-chemnitz.de/~potts/nfft/download/$source.tar.gz
fi

if [ ! -e $source ]
then
    tar xzvf $source.tar.gz
fi



cd $source
./configure --prefix=${PREFIX}
make
make install

echo "================================================================"
echo "To have access to libnfft, please add the following line to your"
echo "shell configuration file (for example .bashrc):"
echo ""
echo "   export LD_LIBRARY_PATH=\"${PREFIX}/lib:\$LD_LIBRARY_PATH\""
echo "================================================================"

#echo "export LD_LIBRARY_PATH=${PREFIX}/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

cd $bak
cd ../sandbox/nfft/
NFFT_DIR=${PREFIX} python setup.py build_ext --inplace
