# This Dockerfile is used only for Binder only. For other
# Dockerfiles, please go to install/docker directory.

FROM finmag/finmag:dependencies

# Tidy up the base image.
RUN rmdir /io/finmag

# Clone the finmag repository.
WORKDIR /
RUN git clone https://github.com/fangohr/finmag.git

# Pre-compile finmag.
WORKDIR /finmag/native
RUN make

ENV PYTHONPATH /finmag/src

# Commands to make Binder work.
RUN pip install --no-cache-dir notebook==5.*
ENV NB_USER finmaguser
ENV NB_UID 1000
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