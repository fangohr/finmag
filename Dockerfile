# This Dockerfile is used for binder only. For other Dockerfiles, please go to
# install/docker directory.

FROM finmag/finmag:latest

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