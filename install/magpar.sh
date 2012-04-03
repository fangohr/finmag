#!/bin/bash

PREFIX="$HOME" # EDIT HERE.

source=magpar-0.9

cp magpar.patch $PREFIX
cp magpar_code.patch $PREFIX
cd $PREFIX
if [ ! -e $source.tar.gz ]
then
   wget http://www.magpar.net/static/$source/download/$source.tar.gz
fi

if [ ! -e $source ]
then
    tar xzvf $source.tar.gz
fi

mv magpar.patch $source/src
mv magpar_code.patch $source/src

cd $source/src
patch -p1 < magpar.patch
patch -p1 < magpar_code.patch

make -f Makefile.libs
make

echo "Please add magpar-0.9/src/magpar.exe to your path."
