import numpy as np
import dolfin as df

mesh = df.UnitCubeMesh(2,2,2)
mesh.init()

boundary = df.DomainBoundary()
facet_is_boundary = df.FacetFunction("bool", mesh)
facet_is_boundary.set_all(False)
boundary.mark(facet_is_boundary, True)

boundary_facets = [] # shape is n, 3, 3 we will want 3 * 3 * n for csa
boundary_vertices_indices = set() # helps us avoid duplicates
boundary_vertices = [] # shape is m, 3

for facet in df.facets(mesh):
    if facet_is_boundary[facet.index()]:
        triangle = []
        for vertex in df.vertices(facet):
            if not vertex.index() in boundary_vertices_indices:
                boundary_vertices_indices.add(vertex.index())
                boundary_vertices.append(mesh.coordinates()[vertex.index()])
            triangle.append(mesh.coordinates()[vertex.index()])
        boundary_facets.append(triangle)

boundary_facets = np.array(boundary_facets)
boundary_vertices = np.array(boundary_vertices)

print "{} of {} facets are part of the boundary.".format(
        boundary_facets.shape[0], mesh.num_facets())
print "{} of {} vertices are part of the boundary.".format(
        boundary_vertices.shape[0], mesh.num_vertices())

# TODO: massage the data, so we can call the solid angle computation with it.
