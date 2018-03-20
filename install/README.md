Overview
--------

A major challenge in installing FinMag are the binary dependencies; in
particular on FEniCS. The FEniCS API has changed occasionally, and
the finmag project has not always kept up with these changes. As a
result, it depends on a particular version of FEniCS. Through some
mistake, that version has disappearend from dpkg servers. However,
there is a Docker image that provides these base libraries, and the
most effective way to work with Finmag is to install it inside a
(Docker) container that provides the right FEniCS version.

This is what the recommended Docker Image (see [Readme](https://github.com/fangohr/finmag/blob/master/README.md))
provides.

In this subdirectory and subfolders, there are snippets from various
attempts to install finmag, and as they have changed over the years.




Installation
------------

Finmag dependencies can be installed by running an appropriate script
(for the Ubuntu version you use) in `install/` directory or by making
a Docker container with Dockerfile in `install/docker/`.

If you decide to install dependencies using a shell script in
`install/` directory, you will need to add the path to the
`finmag/src/` directory to your $PYTHONPATH.

