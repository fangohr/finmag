import numpy as np
import dolfin as df
from aeon import timed, mtimed, Timer, default_timer
from finmag.util.consts import mu0
from finmag.native.llg import compute_bem_fk
from finmag.util.meshes import nodal_volume
from finmag.util import helpers
from fk_demag import FKDemag



class Demag2D(FKDemag):
    """
    To compute the demagnetisation field on a 2d mesh using the normal 
    Fredkin-Koehler method, the idea is to construct a 3d mesh based on 
    the given 2d mesh.
    """
    def __init__(self, name='Demag2D', thickness=1, thin_film=False):

        self.name = name
        
        self.thickness = thickness

        super(Demag2D, self).__init__(name=name, thin_film=thin_film)

    def create_3d_mesh(self, mesh):
        
        nv = mesh.num_vertices()
        nc = mesh.num_cells()
        h = self.thickness

        mesh3 = df.Mesh()
        editor = df.MeshEditor()
        editor.open(mesh3, 3, 3)
        editor.init_vertices(2*nv)
        editor.init_cells(3*nc)

        for v in df.vertices(mesh):
            i = v.index()
            p = v.point()
            editor.add_vertex(i, p.x(),p.y(),0)
            editor.add_vertex(i+nv, p.x(),p.y(),h)

        gid = 0
        for c in df.cells(mesh):
            i,j,k = c.entities(0)
            editor.add_cell(gid, i, j, k, i+nv)
            gid = gid + 1
            editor.add_cell(gid, j, j+nv, k, i+nv)
            gid = gid + 1
            editor.add_cell(gid, k, k+nv, j+nv, i+nv)
            gid = gid + 1

        editor.close()
        return mesh3

    def build_mapping(self, S3, V3):
        """
        S3 is the vector function space of the 2d mesh
        V3 is the vector function space of the corresponding 3d mesh
        """
        vert_to_dof2 = df.vertex_to_dof_map(S3)
        dof_to_vert2 = df.dof_to_vertex_map(S3)

        vert_to_dof3 = df.vertex_to_dof_map(V3)
        dof_to_vert3 = df.dof_to_vertex_map(V3)

        map_2d_to_3d = np.zeros(S3.dim(), dtype=np.int32)

        for i in range(S3.dim()):
            map_2d_to_3d[i] = vert_to_dof3[dof_to_vert2[i]]

        self.map_2d_to_3d = map_2d_to_3d
        #print map_2d_to_3d

        map_3d_to_2d = np.zeros(V3.dim(), dtype=np.int32)
        for i in range(V3.dim()):
            map_3d_to_2d[i] =  vert_to_dof2[dof_to_vert3[i]%S3.dim()]

        self.map_3d_to_2d = map_3d_to_2d
        #print map_3d_to_2d 


    def setup(self, S3, m, Ms, unit_length=1):
        """
        Setup the FKDemag instance. Usually called automatically by the Simulation object.

        *Arguments*

        S3: dolfin.VectorFunctionSpace

            The finite element space the magnetisation is defined on.

        m: dolfin.Function on S3

            The unit magnetisation.

        Ms: float

            The saturation magnetisation in A/m.

        unit_length: float

            The length (in m) represented by one unit on the mesh. Default 1.

        """
        self.short_m = m
        self.Ms = Ms
        self.unit_length = unit_length

        mesh = S3.mesh()
        mesh3 = self.create_3d_mesh(mesh)

        V1 = df.FunctionSpace(mesh3, "Lagrange", 1)
        V3 = df.VectorFunctionSpace(mesh3, "Lagrange", 1)

        mm  = df.Function(V3)

        self.build_mapping(S3,V3)

        super(Demag2D, self).setup(V3, mm, Ms, unit_length)

    
    def compute_field(self):
        """
        Compute the demagnetising field.

        *Returns*
            numpy.ndarray
                The demagnetising field.

        """
        self.m.vector().set_local(self.short_m.vector().array()[self.map_3d_to_2d])
        self._compute_magnetic_potential()

        return self._compute_gradient()[self.map_2d_to_3d]