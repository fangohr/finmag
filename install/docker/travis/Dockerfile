FROM finmag/finmag:dependencies

# Tidy up the base image.
RUN rmdir /io/finmag

# Copy the finmag repository.
COPY . /finmag/

RUN ls -al

# Pre-compile finmag.
WORKDIR /finmag/native
RUN make

WORKDIR /finmag

ENV PYTHONPATH /finmag/src
