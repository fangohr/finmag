FROM finmag/finmag:dependencies

# Tidy up the base image.
RUN rmdir /io/finmag

# Clone the finmag repository.
WORKDIR /
RUN git clone https://github.com/fangohr/finmag.git

# Pre-compile finmag.
WORKDIR /finmag/native
RUN make

# Make sure the files created on host are not owned by root.
RUN adduser finmaguser
RUN chown -R finmaguser /io
RUN chown -R finmaguser /finmag

USER finmaguser

WORKDIR /io

ENV PYTHONPATH /finmag/src
