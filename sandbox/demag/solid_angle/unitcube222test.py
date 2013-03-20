from dolfin import *
from finmag.native.llg import compute_solid_angle
import numpy as np

####
# This is the output from info(mesh, True) under connectivity 2 - 1.
# Keys - Triangle number
# Values - Node number
tmpdic = { \
      0: [1, 4, 13],
      1: [0, 4, 13],
      2: [0, 1, 13],
      3: [0, 1, 4],
      4: [1, 10, 13],
      5: [0, 10, 13],
      6: [0, 1, 10],
      7: [9, 10, 13],
      8: [0, 9, 13],
      9: [0, 9, 10],
      10: [3, 4, 13],
      11: [0, 3, 13],
      12: [0, 3, 4],
      13: [9, 12, 13],
      14: [0, 12, 13],
      15: [0, 9, 12],
      16: [3, 12, 13],
      17: [0, 3, 12],
      18: [2, 5, 14],
      19: [1, 5, 14],
      20: [1, 2, 14],
      21: [1, 2, 5],
      22: [2, 11, 14],
      23: [1, 11, 14],
      24: [1, 2, 11],
      25: [10, 11, 14],
      26: [1, 10, 14],
      27: [1, 10, 11],
      28: [4, 5, 14],
      29: [1, 4, 14],
      30: [1, 4, 5],
      31: [10, 13, 14],
      32: [1, 13, 14],
      33: [4, 13, 14],
      34: [4, 7, 16],
      35: [3, 7, 16],
      36: [3, 4, 16],
      37: [3, 4, 7],
      38: [4, 13, 16],
      39: [3, 13, 16],
      40: [12, 13, 16],
      41: [3, 12, 16],
      42: [6, 7, 16],
      43: [3, 6, 16],
      44: [3, 6, 7],
      45: [12, 15, 16],
      46: [3, 15, 16],
      47: [3, 12, 15],
      48: [6, 15, 16],
      49: [3, 6, 15],
      50: [5, 8, 17],
      51: [4, 8, 17],
      52: [4, 5, 17],
      53: [4, 5, 8],
      54: [5, 14, 17],
      55: [4, 14, 17],
      56: [13, 14, 17],
      57: [4, 13, 17],
      58: [7, 8, 17],
      59: [4, 7, 17],
      60: [4, 7, 8],
      61: [13, 16, 17],
      62: [4, 16, 17],
      63: [7, 16, 17],
      64: [10, 13, 22],
      65: [9, 13, 22],
      66: [9, 10, 22],
      67: [10, 19, 22],
      68: [9, 19, 22],
      69: [9, 10, 19],
      70: [18, 19, 22],
      71: [9, 18, 22],
      72: [9, 18, 19],
      73: [12, 13, 22],
      74: [9, 12, 22],
      75: [18, 21, 22],
      76: [9, 21, 22],
      77: [9, 18, 21],
      78: [12, 21, 22],
      79: [9, 12, 21],
      80: [11, 14, 23],
      81: [10, 14, 23],
      82: [10, 11, 23],
      83: [11, 20, 23],
      84: [10, 20, 23],
      85: [10, 11, 20],
      86: [19, 20, 23],
      87: [10, 19, 23],
      88: [10, 19, 20],
      89: [13, 14, 23],
      90: [10, 13, 23],
      91: [19, 22, 23],
      92: [10, 22, 23],
      93: [13, 22, 23],
      94: [13, 16, 25],
      95: [12, 16, 25],
      96: [12, 13, 25],
      97: [13, 22, 25],
      98: [12, 22, 25],
      99: [21, 22, 25],
      100: [12, 21, 25],
      101: [15, 16, 25],
      102: [12, 15, 25],
      103: [21, 24, 25],
      104: [12, 24, 25],
      105: [12, 21, 24],
      106: [15, 24, 25],
      107: [12, 15, 24],
      108: [14, 17, 26],
      109: [13, 17, 26],
      110: [13, 14, 26],
      111: [14, 23, 26],
      112: [13, 23, 26],
      113: [22, 23, 26],
      114: [13, 22, 26],
      115: [16, 17, 26],
      116: [13, 16, 26],
      117: [22, 25, 26],
      118: [13, 25, 26],
      119: [16, 25, 26]}
####
# The above dictionary is the output from info(mesh, True) under connectivity 2 - 1.
# Keys - Triangle number
# Values - Node number


mesh = UnitCubeMesh(2,2,2)
V = FunctionSpace(mesh, "CG", 1)
mesh.init()
d = mesh.topology().dim()
dm = V.dofmap()



####
# Gabriel's code to make a dictionary with coordinates of nodes on the boundary.
# Keys - Node number
# Values - Nodal coordinates
#
boundarydofs = DirichletBC(V,0,"on_boundary").get_boundary_values()
#Build a dictionary with all dofs and their coordinates
#TODO Optimize me!
doftionary = {}
for facet in facets(mesh):
    cells = facet.entities(d)
    if len(cells) == 2:
        continue
    elif len(cells) == 1:
        #create one cell (since we have CG)
        cell = Cell(mesh, cells[0])
        #Create the cell dofs and see if any
        #of the global numbers turn up in BoundaryDofs
        #If so update the BoundaryDic with the coordinates
        celldofcord = dm.tabulate_coordinates(cell)
        globaldofs = dm.cell_dofs(cells[0])
        globalcoord = dm.tabulate_coordinates(cell)

        for locind,dof in enumerate(globaldofs):
            doftionary[dof] = globalcoord[locind]
        #restrict the doftionary to the boundary
        for x in [x for x in doftionary if x not in boundarydofs]:
            doftionary.pop(x)
    else:
        assert 1==2,"Expected only two cells per facet and not " + str(len(cells))


# Create a 3 x m array with the coordinates of each node on the boundary
dofs = doftionary.keys()
m = len(dofs)
dofarr = np.zeros((3, m))
for i, dof in enumerate(doftionary.iterkeys()):
    dofarr[:,i] = doftionary[dof]


# Create a 3 x 3 x n array with coordinates of each corner 
# of the triangle for each triangle in the mesh.
coor = mesh.coordinates()
n = len(tmpdic.keys())
arr = np.zeros((3,3,n))
for i, value in tmpdic.iteritems():
    arr[:,0,i] = coor[value[0]]
    arr[:,1,i] = coor[value[1]]
    arr[:,2,i] = coor[value[2]]

print dofarr
print arr
output = np.zeros(m)
compute_solid_angle(dofarr, arr, output)
print output
