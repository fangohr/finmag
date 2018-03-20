import dolfin as df

#
# The mesh contains two regions.
#   region 1: A disk (cylinder).
#   region 2: The cuboid around the disk.
#

mesh = df.Mesh("embedded_disk.xml.gz")
regions = df.MeshFunction("uint", mesh, "embedded_disk_mat.xml")

#
# Example 1
# Compute the volume of the regions and the total mesh.
#

dxx = df.dx[regions]  # is this following "best practices" ?

volume_disk = df.assemble(df.Constant(1) * dxx(1), mesh=mesh)
volume_cuboid = df.assemble(df.Constant(1) * dxx(2), mesh=mesh)

print "Volume of the embedded disk: v1 = {}.".format(volume_disk)
print "Volume of the medium around it: v2 = {}.".format(volume_cuboid)

volume_total = df.assemble(df.Constant(1) * df.dx, mesh=mesh)
print "Total volume: vtot = {}. Same as v1 + v2 = {}.".format(
    volume_total, volume_disk + volume_cuboid)

#
# Example 2
# Have a material value that is different on the two regions.
#

k_values = [10, 100]

DG0 = df.FunctionSpace(mesh, "DG", 0)
k = df.Function(DG0)

# Will this work with the new ordering introduced in dolfin-1.1.0?
# (dofs ordering won't correspond to geometry anymore)
for cell_no, region_no in enumerate(regions.array()):
    k.vector()[cell_no] = k_values[region_no - 1]

#
# Example 3
# FunctionSpaces defined on regions instead of the whole mesh.
#

disk_mesh = df.SubMesh(mesh, regions, 1)
CG1_disk = df.FunctionSpace(disk_mesh, "Lagrange", 1)

cuboid_mesh = df.SubMesh(mesh, regions, 2)
CG1_cuboid = df.FunctionSpace(cuboid_mesh, "Lagrange", 1)

#
# Visualise the different regions and the values of k
#

k_CG1 = df.interpolate(k, df.FunctionSpace(mesh, "CG", 1))
k_disk = df.project(k, CG1_disk)
k_cuboid = df.project(k, CG1_cuboid)

df.common.plotting.Viper.write_png(df.plot(k), "k_DG0.png")
df.common.plotting.Viper.write_png(df.plot(k_CG1), "k_interpolated_CG1.png")
df.common.plotting.Viper.write_png(df.plot(k_disk), "k_projected_disk.png")
df.common.plotting.Viper.write_png(df.plot(k_cuboid), "k_projected_cuboid.png")
