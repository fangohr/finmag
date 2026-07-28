FROM finmag/finmag:dependencies

# Bootstrap image for Python 3 porting work.
#
# Important: the legacy Ubuntu/FEniCS packaging used by finmag provides
# python-dolfin for the 2017.1.0 stack, but not python3-dolfin. This image
# is therefore not a full Python 3 runtime for Finmag yet. Its purpose is to
# support syntax checks, packaging/import cleanup, and other Python 3 work
# that can proceed before a Python 3 DOLFIN build is available.

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel pytest future

WORKDIR /finmag

ENV PYTHONPATH=/finmag/src

CMD ["/bin/bash"]
