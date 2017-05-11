# Singularity Container for Finmag

To use:

* Build your own container

Locally you need to compile and install [Singularity](http://singularity.lbl.gov/). I've only tested this with v2.2.1 which as of 03/05/2017
is what is also installed on Iridis 4. However, you should make sure you use the same version as is currently installed

Once you have it installed, just run 'make' to build the container.

* Test it

Copy the container to Iridis, and try out some of the following commands

> singularity exec container.img python -c 'import fenics';
> singularity shell container.img

* To run Finmag, you need to download this locally - your home directory is shared into the container, so
it doesn't matter that it's not in the filesystem of the container. Set your PYTHONPATH variable to point
to /path/to/finmag/src, and then run:

> singularity exec container.img python -c 'import finmag'
