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

        map_2d_to_3d = np.zeros(V3.dim(), dtype=np.int32)

        n2d  = S3.dim()
        for i in range(n2d):
            map_2d_to_3d[i] = vert_to_dof3[dof_to_vert2[i]]
            map_2d_to_3d[i+n2d] = vert_to_dof3[dof_to_vert2[i]+n2d]

        self.map_2d_to_3d = map_2d_to_3d
        #print map_2d_to_3d

        n3d = V3.dim()
        map_3d_to_2d = np.zeros(V3.dim(), dtype=np.int32)
        for i in range(V3.dim()):
            map_3d_to_2d[i] =  vert_to_dof2[dof_to_vert3[i]%n2d]

        self.map_3d_to_2d = map_3d_to_2d
        #print map_3d_to_2d 

    def create_dg3_from_dg2(self, mesh, dg2):

        dg3 = df.FunctionSpace(mesh,'DG',0)

        class HelperExpression(df.Expression):
            def __init__(self,value):
                super(HelperExpression, self).__init__()
                self.fun = value

            def eval(self, value, x):
                value[0] = self.fun((x[0],x[1]))

        hexp = HelperExpression(dg2)
        fun = df.interpolate(hexp, dg3)

        return fun

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
        self.unit_length = unit_length

        mesh = S3.mesh()
        mesh3 = self.create_3d_mesh(mesh)

        V1 = df.FunctionSpace(mesh3, "Lagrange", 1)
        V3 = df.VectorFunctionSpace(mesh3, "Lagrange", 1)

        mm  = df.Function(V3)

        self.build_mapping(S3,V3)

        Ms_dg3 = self.create_dg3_from_dg2(mesh3, Ms)

        super(Demag2D, self).setup(V3, mm, Ms_dg3, unit_length)

    def compute_energy(self):
        """
        Compute the total energy of the field.

        .. math::

            E_\\mathrm{d} = -\\frac12 \\mu_0 \\int_\\Omega
            H_\\mathrm{d} \\cdot \\vec M \\mathrm{d}x

        *Returns*
            Float
                The energy of the demagnetising field.

        """
        self._H_func.vector()[:] = self.__compute_field()
        return df.assemble(self._E) * self.unit_length ** self.dim

    @mtimed(default_timer)
    def energy_density(self):
        """
        Compute the energy density in the field.

        .. math::
            \\rho = \\frac{E_{\\mathrm{d}, i}}{V_i},

        where V_i is the volume associated with the node i.

        *Returns*
            numpy.ndarray
                The energy density of the demagnetising field.

        """
        self._H_func.vector()[:] = self.__compute_field()
        nodal_E = df.assemble(self._nodal_E).array() * self.unit_length ** self.dim
        return nodal_E / self._nodal_volumes

    def __compute_field(self):
        self.m.vector().set_local(self.short_m.vector().array()[self.map_3d_to_2d])
        self._compute_magnetic_potential()
        return self._compute_gradient()

    
    def compute_field(self):
        """
        Compute the demagnetising field.

        *Returns*
            numpy.ndarray
                The demagnetising field.

        """
        f = self.__compute_field()[self.map_2d_to_3d]
        f.shape = (2,-1)
        f_avg = (f[0]+f[1])/2.0
        f.shape=(-1,)
        return f_avg