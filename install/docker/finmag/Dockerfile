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
    libslepc3.6.1-dev gmsh
RUN PETSC_DIR=/usr/lib/petsc SLEPC_DIR=/usr/lib/slepc pip install https://bitbucket.org/slepc/slepc4py/downloads/slepc4py-3.6.0.tar.gz --allow-all-external
RUN pip install --upgrade pip aeon sh future git+https://github.com/fangohr/dolfinh5tools.git

# Not necessary convenience packages (can be removed)!!!
RUN apt-get install -y emacs
RUN pip install --upgrade pytest

# Install the newest netgen (to avoid segmentation fault)
WORKDIR /usr/local
RUN apt-get install -y wget autoconf tk8.5-dev tcl8.5-dev libxmu-dev gawk python3-tk python3-pip metis openmpi-bin libopenmpi-dev libtogl-dev python3-dev
RUN wget http://kent.dl.sourceforge.net/project/netgen-mesher/netgen-mesher/5.3/netgen-5.3.1.tar.gz 
RUN tar -xvf netgen-5.3.1.tar.gz
WORKDIR /usr/local/netgen-5.3.1
RUN autoconf
RUN ./configure --with-sysroot=/usr/lib/ --with-tcl=/usr/lib/tcl8.5/ --with-tk=/usr/lib/tk8.5 --with-togl=/usr/lib/ --with-metis=/usr/lib/x86_64-linux-gnu/ --enable-nglib
RUN make
RUN make install
WORKDIR /root
ENV PATH /opt/netgen/bin:$PATH
ENV NETGENDIR /opt/netgen/bin


# install buildbot
RUN pip install --upgrade buildbot buildbot-console-view buildbot-waterfall-view buildbot-worker buildbot-www

# create working directory, with the intention to mount /io from host.
# We also expect that a finmag repository is checked out to /io/finmag
ENV PYTHONPATH=/io/finmag/src

RUN mkdir /io
WORKDIR /io


# not mounted yet
# RUN cd /io/finmag/native && make




