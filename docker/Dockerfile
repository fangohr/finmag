FROM ubuntu:16.04

COPY 99apt_behave /etc/apt/apt.conf.d/
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install software-properties-common

RUN add-apt-repository -y ppa:fenics-packages/fenics
RUN apt-get update
RUN apt-get install openssh-client software-properties-common python-m2crypto \
    python-netifaces secure-delete swig python-pip python-matplotlib python-visual \
    python-scipy python-setuptools python-progressbar cython netgen netgen-doc \
    xpra paraview-python libsundials-serial-dev libboost-python-dev sudo \
    libboost-test-dev git
# From the fenics PPA
RUN apt-get install fenics python-dolfin dolfin-bin libdolfin-dev

RUN pip install --upgrade setuptools pip
RUN pip install sphinx pytest sh diff-match-patch aeon mpi4py \
    ipython pyzmq tornado future
RUN pip install "git+https://github.com/fangohr/dolfinh5tools#egg=dolfinh5tools"

ENV FINMAG_HOME /home/finmag

# Set up user so that we do not run as root
RUN useradd -m -s /bin/bash -G sudo finmag && \
    echo "finmag:docker" | chpasswd && \
    echo "finmag ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
RUN chown -R finmag $FINMAG_HOME

# For bind mounts from host
RUN mkdir /io
RUN chown -R finmag /io

USER finmag
RUN touch $FINMAG_HOME/.sudo_as_admin_successful

WORKDIR /io
CMD ["/bin/bash","-i"]
