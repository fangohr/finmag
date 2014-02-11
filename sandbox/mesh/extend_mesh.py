import dolfin as df

def test_triangle_mesh():

        # Create mesh object and open editor
	mesh = df.Mesh()
	editor = df.MeshEditor()
	editor.open(mesh, 2, 2)
	editor.init_vertices(3)
	editor.init_cells(1)

        # Add vertices
	editor.add_vertex(0, 0.0, 0.0)
	editor.add_vertex(1, 1.0, 0.0)
	editor.add_vertex(2, 0.0, 1.0)

        # Add cell
	editor.add_cell(0, 0, 1, 2)

        # Close editor
	editor.close()

	return mesh


def test_triangle_mesh2():

        # Create mesh object and open editor
	mesh = df.Mesh()
	editor = df.MeshEditor()
	editor.open(mesh, 3, 3)
	editor.init_vertices(4)
	editor.init_cells(1)

	# Add vertices
	editor.add_vertex(0, 0.0, 0.0, 0.0)
	editor.add_vertex(1, 1.0, 0.0, 0.0)
	editor.add_vertex(2, 0.0, 1.0, 0.0)
	editor.add_vertex(3, 0.0, 0.0, 1.0)


        # Add cell
	editor.add_cell(0, 0, 1, 2, 3)

        # Close editor
	editor.close()

	return mesh


def create_3d_mesh(mesh, h=1):
	assert mesh.topology().dim() == 2
	for cell in df.cells(mesh):
		print cell
		print cell.entities(0)
		print cell.get_vertex_coordinates()

	nv = mesh.num_vertices()
	nc = mesh.num_cells()

	mesh3 = df.Mesh()
	editor = df.MeshEditor()
	editor.open(mesh3, 3, 3)
	editor.init_vertices(2*nv)
	editor.init_cells(3*nc)

	for v in df.vertices(mesh):
		i = v.global_index()
		p = v.point()
		editor.add_vertex(i, p.x(),p.y(),0)
		editor.add_vertex(i+nv, p.x(),p.y(),h)

	gid = 0
	for c in df.cells(mesh):
		#gid = c.global_index()
		i,j,k = c.entities(0)
		print i,j,k
		editor.add_cell(gid, i, j, k, i+nv)
		gid = gid + 1
		editor.add_cell(gid, j, j+nv, k, i+nv)
		gid = gid + 1
		editor.add_cell(gid, k, k+nv, j+nv, i+nv)
		gid = gid + 1

	editor.close()
	return mesh3






mesh = df.UnitSquareMesh(2,2)
mesh = df.UnitTriangleMesh()
tri = create_3d_mesh(mesh)

print tri.coordinates()

df.plot(tri)

df.interactive()