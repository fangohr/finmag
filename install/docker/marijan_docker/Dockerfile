FROM ubuntu:16.04

RUN apt-get  update -y

# Install fenics
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:fenics-packages/fenics
RUN apt-get update -y
RUN apt-get install -y fenics --allow-unauthenticated --no-install-recommends
RUN apt-get dist-upgrade -y

RUN apt-get install -y git python-pip gmsh netgen cython libsundials-serial-dev \
    libboost-python-dev libboost-test-dev ipython python-scipy python-matplotlib

RUN pip install --upgrade pip aeon future sh pytest \
    git+https://github.com/fangohr/dolfinh5tools.git

# Install the newest netgen (to avoid segmentation fault)
WORKDIR /usr/local
RUN apt-get install -y wget autoconf tk8.5-dev tcl8.5-dev libxmu-dev gawk python3-tk \
    python3-pip metis openmpi-bin libopenmpi-dev libtogl-dev python3-dev python-psutil \
    --no-install-recommends
RUN wget http://kent.dl.sourceforge.net/project/netgen-mesher/netgen-mesher/5.3/netgen-5.3.1.tar.gz
RUN tar -xvf netgen-5.3.1.tar.gz
WORKDIR /usr/local/netgen-5.3.1
RUN autoconf
RUN ./configure --with-sysroot=/usr/lib/ --with-tcl=/usr/lib/tcl8.5/ --with-tk=/usr/lib/tk8.5 --with-togl=/usr/lib/ --with-metis=/usr/lib/x86_64-linux-gnu/ --enable-nglib
RUN make
RUN make install

ENV PATH /opt/netgen/bin:$PATH
ENV NETGENDIR /opt/netgen/bin

RUN mkdir -p /io/finmag

#ENV PYTHONPATH=/root/finmag/src
ENV PYTHONPATH=/io/finmag/src
WORKDIR /io/finmag
