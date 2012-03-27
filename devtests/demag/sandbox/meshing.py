#Test of Meshing tools

from dolfin import *
from finmag.util.convert_mesh import convert_mesh
from finmag.demag.problems.prob_base import FemBemDeMagProblem 

mesh = Mesh(convert_mesh("sphere10.geo"))
##plot(mesh)
##interactive()

class GoodMesh(FemBemDeMagProblem):
    def __init__(self):
        super(GoodMesh,self).__init__(mesh,("1","0","0"))

