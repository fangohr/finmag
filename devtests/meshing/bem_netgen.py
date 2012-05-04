import dolfin as df
from finmag.native.llg import OrientedBoundaryMesh, compute_bem

print "Building bem with dolfin mesh"
mesh = df.Box(0,0,0,30,30,100,3,3,10)
bem, b2g_map = compute_bem(OrientedBoundaryMesh(mesh))
print "That went okay.."

print "Building bem with netgen mesh converted via dolfin-convert"
netgen_mesh = df.Mesh("bar30_30_100.xml.gz")
bem, b2g_map = compute_bem(OrientedBoundaryMesh(netgen_mesh))

