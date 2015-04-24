#!/usr/bin/env python
# 2015-04-24, Created by H Fuchs <code@hfuchs.net>
# This does nothing more than installing the already-created
# finmag-tree.

from distutils.core import setup

setup(name = 'Finmag',
        version = '1.0',
        description = 'Finmag',
        author = 'Hans Fangohr et al',
        author_email = 'fangohr@soton.ac.uk',
        package_dir = {'': '/tmp/finmag/'},
        packages = [
            'finmag',
            'finmag.scheduler',
            'finmag.example.normal_modes',
            'finmag.example',
            'finmag.drivers',
            'finmag.normal_modes.eigenmodes',
            'finmag.normal_modes',
            'finmag.normal_modes.deprecated',
            'finmag.energies',
            'finmag.energies.demag',
            'finmag.tests.jacobean',
            'finmag.tests',
            'finmag.tests.demag',
            'finmag.util',
            'finmag.util.ode',
            'finmag.util.ode.tests',
            'finmag.util.tests',
            'finmag.util.oommf',
            'finmag.native',
            'finmag.sim',
            'finmag.physics',
            'finmag.physics.llb',
            'finmag.physics.llb.tests'
            ]
        )
