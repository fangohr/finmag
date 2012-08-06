#!/bin/bash

#needs gfortran installed
sudo apt-get install libfftw3-dev

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
./configure
make
sudo make install

echo "export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

cd $bak
cd ../devtests/nfft/
python setup.py build_ext --inplace
