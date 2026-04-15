FROM debian:stretch

# Experimental Python 3 runtime for the old DOLFIN stack used by Finmag.
#
# Key findings behind this image:
# - upstream DOLFIN 2017.1.0 switched the default build to Python 3;
# - Debian sid snapshot 2017-09-20 contains python3-dolfin=2017.1.0-4 and
#   matching python3-ffc / python3-ufl / python3-dijitso / python3-fiat /
#   python3-instant packages;
# - the exact package versions were discovered by probing the snapshot inside
#   disposable containers from this repo.
#
# This image is intentionally isolated from the current Finmag Docker images.
# The legacy finmag images are Ubuntu/xenial based and centered on Python 2.
# Here we use the historically matching Debian snapshot to get a coherent
# Python 3 DOLFIN 2017 environment first.

RUN cat > /etc/apt/sources.list <<'EOF'
deb http://snapshot.debian.org/archive/debian/20170920T000000Z sid main
deb-src http://snapshot.debian.org/archive/debian/20170920T000000Z sid main
EOF

RUN printf 'Acquire::Check-Valid-Until "false";\nAcquire::AllowInsecureRepositories "true";\nAPT::Get::AllowUnauthenticated "true";\n' > /etc/apt/apt.conf.d/99snapshot

# Move the base system onto the historical snapshot so that the exact DOLFIN
# dependency set can be resolved consistently.
RUN apt-get update -y && \
    apt-get dist-upgrade -y --allow-unauthenticated

RUN apt-get install -y --no-install-recommends --allow-unauthenticated --allow-downgrades \
    python3 \
    python3-dev \
    python3-matplotlib \
    python3-pip \
    python3-scipy \
    python3-setuptools \
    default-libmysqlclient-dev \
    cython3 \
    libboost-python-dev \
    libboost-test-dev \
    libfreetype6-dev \
    libgdal-dev \
    libhdf5-openmpi-dev \
    libhdf5-mpi-dev \
    libnetcdf-dev \
    netgen \
    libpng-dev \
    libptscotch-dev \
    libscotch-dev \
    libsundials-dev=2.7.0+dfsg-2 \
    libtiff-dev \
    libvtk6-dev \
    libxft-dev \
    libpetsc3.7.6-dev \
    libslepc3.7.4-dev \
    tcl-dev \
    tix \
    tk-dev \
    xauth \
    xvfb \
    zlib1g=1:1.2.8.dfsg-5 \
    zlib1g-dev \
    libdolfin-dev=2017.1.0-4 \
    python3-dolfin=2017.1.0-4 \
    python3-ffc=2017.1.0-2 \
    python3-ufl=2017.1.0-2 \
    python3-dijitso=2017.1.0-2 \
    python3-fiat=2017.1.0-2 \
    python3-instant=2017.1.0-2 \
    python3-petsc4py=3.7.0-3+b2 \
    python3-slepc4py=3.7.0-3+b1

# Netgen 4.9.13 in this snapshot links against Tcl/Tk 8.5 and looks for Tix in
# historical 8.5-era locations, while Debian's `tix` package installs the files
# under the shared 8.6 paths. Provide the compatibility links it expects.
RUN ln -s /usr/share/tcltk/tcl8.6/Tix8.4 /usr/share/tcltk/tcl8.5/Tix8.4 && \
    ln -s /usr/share/tcltk/tcl8.6/Tix8.4 /usr/share/tcltk/Tix8.4 && \
    ln -s /usr/share/tcltk/tcl8.6/Tix8.4 /usr/lib/Tix8.4

RUN pip3 install --no-cache-dir pytest==3.2.1

RUN OMPI_MCA_plm=isolated python3 - <<'PY'
import dolfin
import sys
print(sys.version)
print(dolfin.__version__)
print(dolfin.__file__)
PY

WORKDIR /io

ENV PYTHONPATH=/io/finmag/src

CMD ["/bin/bash"]
