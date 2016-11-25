FROM ubuntu:16.04

RUN apt-get update -y

# Install fenics
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:fenics-packages/fenics
RUN apt-get update -y
RUN apt-get install -y fenics
RUN apt-get dist-upgrade -y

# Packages required to build finmag
RUN apt-get install -y cython libsundials-serial-dev libboost-python-dev \
    libboost-test-dev

# Packages required to run tests
RUN apt-get install -y python-pip git libpetsc3.6 libpetsc3.6.2-dev libslepc3.6.1 \
    libslepc3.6.1-dev netgen gmsh
RUN PETSC_DIR=/usr/lib/petsc SLEPC_DIR=/usr/lib/slepc pip install https://bitbucket.org/slepc/slepc4py/downloads/slepc4py-3.6.0.tar.gz --allow-all-external
RUN pip install --upgrade pip aeon sh future git+https://github.com/fangohr/dolfinh5tools.git

# Not necessary convenience packages (can be removed)!!!
RUN pip install --upgrade pytest
RUN apt-get install -y emacs