from dolfin import *

from common import mesh
F = HDF5File('meshfile_serial.h5', 'w')
F.write(mesh, 'mymesh')
