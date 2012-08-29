import dolfin as df
from finmag.util.meshes import from_geofile

mesh = from_geofile("film.geo")
print mesh
df.plot(mesh)
df.interactive()
