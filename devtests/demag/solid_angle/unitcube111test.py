import dolfin as df

mesh = df.UnitCube(2,2,2)
mesh.init()

"""
class Boundary(df.SubDomain):
    def inside(self, x, on_boundary):
        print on_boundary
        return on_boundary
boundary_subdomain = Boundary()

face_meshfunction = dolfin.MeshFunction('uint', mesh, 2)
face_meshfunction.set_all(0)
boundary_subdomain.mark(face_meshfunction, 1)

"""

boundary = df.DomainBoundary()
face_is_boundary = df.FacetFunction("uint", mesh)
face_is_boundary.set_all(0)
boundary.mark(face_is_boundary, 1)

for face in df.facets(mesh):
    i = face.index()
    if face_is_boundary[i]:
        print "Face {0} is on the boundary.".format(i)
        for vertex in df.vertices(face):
            print vertex.index(), mesh.coordinates()[vertex.index()]
    else:
        print "Face {0} is not on the boundary.".format(i)
