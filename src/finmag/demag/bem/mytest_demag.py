import unittest
import numpy as np
import math
import dolfin as df
import os
from finmag.native.llg import compute_bem_element, compute_bem, OrientedBoundaryMesh
from finmag.util import time_counter
from finmag.sim import helpers

from finmag.demag import belement_magpar
from finmag.sim.llg import LLG

import dolfin as df

mesh = df.UnitSphere(2)
print "llg", np.round(compute_scalar_potential_llg(mesh, m), 3)
print "my", np.round(compute_scalar_potential_native_FK(mesh, m), 3)
