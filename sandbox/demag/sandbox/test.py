"""Function Space restriction Test, how can we map dofs from the
parent mesh to a submesh?"""

from dolfin import *
from solver_nitsche import *

class tester(TruncDeMagSolver):
    def __init__(self):
        mesh = UnitSquareMesh(2,2)
        V = FunctionSpace(mesh,"CG",1)
        u = interpolate(Expression("1 - x[0]"),V)
        
        meshfunc = MeshFunction("uint",mesh,2)
        meshfunc.set_all(0)
        Half().mark(meshfunc,1)
        halfmesh = SubMesh(mesh,meshfunc,1)
        TruncDeMagSolver.__init__(self,mesh)
        print self.restrictfunc
        self.uhalf = self.restrictfunc(u,halfmesh)
##        print "ok"

def orig():
    #Original script that should work
        mesh = UnitSquareMesh(2,2)
        V = FunctionSpace(mesh,"CG",1)
        u = interpolate(Expression("1 - x[0]"),V)
        
        meshfunc = MeshFunction("uint",mesh,2)
        meshfunc.set_all(0)
        Half().mark(meshfunc,1)
        halfmesh = SubMesh(mesh,meshfunc,1)
        
        halfspace = FunctionSpace(halfmesh, "CG", 1)
        
        newmeshfunc = MeshFunction("uint",mesh,2)
        newmeshfunc.set_all(0)

        #This is actually the whole mesh, but compute_vertex_map, only accepts a SubMesh
        wholesubmesh = SubMesh(mesh,newmeshfunc,0)
        map_to_mesh = wholesubmesh.data().mesh_function("parent_vertex_indices")

        #This is a dictionary mapping the matching DOFS from the parent mesh to the SubMesh
        vm = compute_vertex_map(halfmesh,wholesubmesh)

        #Now we want to "restrict" u to the halfspace
        uhalf = Function(halfspace)
        for index,dof in enumerate(uhalf.vector()):
            uhalf.vector()[index] = u.vector()[map_to_mesh[vm[index]]]
        plot(uhalf)
        interactive()
##
test = tester()
print test.uhalf
plot(test.uhalf, title = "Restricted Function")
##plot(u, title = "Whole Function")
interactive()

##orig()
