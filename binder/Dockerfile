# This Dockerfile is used only for Binder only. For other Dockerfiles, please
# go to install/docker directory.

FROM finmag/finmag:dependencies

# Tidy up the base image.
RUN rmdir /io/finmag

# Clone the finmag repository.
WORKDIR /
COPY . /finmag

# Pre-compile finmag.
WORKDIR /finmag/native
RUN make

ENV PYTHONPATH /finmag/src

# Commands to make Binder work.
RUN pip install --no-cache-dir notebook==5.*

ARG NB_USER=finmaguser
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
RUN chown -R ${NB_UID} /finmag
USER ${NB_USER}

WORKDIR /finmag
