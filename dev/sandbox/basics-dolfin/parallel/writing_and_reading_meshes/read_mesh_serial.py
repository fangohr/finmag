import sys
from dolfin import *

try:
    meshfilename = sys.argv[1]
except IndexError:
    print "Error: expecting meshfilename as first argument."
    sys.exit(0)

G = HDF5File(meshfilename, 'r')
mesh2 = Mesh()
G.read(mesh2, 'mymesh')

from common import mesh

def sort_nested(lst, level=1):
    """
    Sort a list of lists up to the given level:

    0 = don't sort
    1 = sort outer list but keep inner lists untouched
    2 = sort inner lists first and then sort outer list
    """
    if level == 0:
        return lst
    elif level == 1:
        return sorted(lst)
    elif level == 2:
        return sorted(map(sorted, lst))
    else:
        raise ValueError("Sorting level must be <= 2")

coords1 = mesh.coordinates().tolist()
coords2 = mesh2.coordinates().tolist()

cells1 = mesh.cells().tolist()
cells2 = mesh2.cells().tolist()

if coords1 == coords2:
    print "Coordinates match."
elif sort_nested(coords1, 1) == sort_nested(coords2, 1):
    print "Coordinates match after sorting outer lists."
elif sort_nested(coords1, 2) == sort_nested(coords2, 2):
    print "Coordinates match after complete sorting."
else:
    print "Coordinates do NOT match."

if cells1 == cells2:
    print "Cells match."
elif sort_nested(cells1, 1) == sort_nested(cells2, 1):
    print "Cells match after sorting outer lists."
elif sort_nested(cells1, 2) == sort_nested(cells2, 2):
    print "Cells match after complete sorting."
else:
    print "Cells do NOT match."
