Overview
--------

This orphan branch contains a BuildBot configuration to test finmag.

Installing BuildBot
-------------------

BuildBot (the software, not the server we are starting) can be installed via
the `pip` of your choice by commanding:

    pip install --upgrade --requirement installation/requirements.txt

If you don't have pip, you can get it from
[here](https://pypi.python.org/pypi/pip/), though you could consider using
[Anaconda](https://www.continuum.io/downloads) or
[Miniconda](http://conda.pydata.org/miniconda.html). You may need to prepend
the command with `sudo ` if you are using the system pip (you'll know if you
get a "Permission denied" OSError).

Get Started
-----------

On Linux, command

    ./start

Then point your browser to `localhost:8010`.