import dolfin as df
import numpy as np


class PeriodicBoundary1D(df.SubDomain):
    """
    Periodic Boundary condition in in x direction
    """
    def __init__(self, mesh):
        super(PeriodicBoundary1D, self).__init__()

        self.mesh = mesh

        self.find_mesh_info()

    def inside(self, x, on_boundary):
        on_x = bool(df.near(x[0], self.xmin) and on_boundary)
        return on_x 

    def map(self, x, y):
        y[0] = x[0] - self.width
        
        if self.dim > 1:
            y[1] = x[1]
        if self.dim > 2:
            y[2] = x[2]

    def find_mesh_info(self):
        coords = self.mesh.coordinates()
        self.length = len(coords)
        max_v = coords.max(axis=0)
        min_v = coords.min(axis=0)

        self.xmin = min_v[0]
        self.xmax = max_v[0]
       
        self.width = self.xmax - self.xmin
        
        self.dim = self.mesh.topology().dim()


class PeriodicBoundary2D(df.SubDomain):
    """
    Periodic Boundary condition in xy-plane.
    """
    def __init__(self, mesh):
        super(PeriodicBoundary2D, self).__init__()

        self.mesh = mesh

        self.find_mesh_info()

    def inside(self, x, on_boundary):
        on_x = bool(df.near(x[0], self.xmin) and x[1] < self.ymax and on_boundary)
        on_y = bool(df.near(x[1], self.ymin) and x[0] < self.xmax and on_boundary)
        return on_x or on_y

    def map(self, x, y):
        y[0] = x[0] - self.width
        y[1] = x[1] - self.height
        
        if self.dim == 3:
            y[2] = x[2]

        if df.near(x[0],self.xmax) and x[1] < self.ymax:
            y[1] = x[1]

        if df.near(x[1],self.ymax) and x[0] < self.xmax:
            y[0] = x[0]

    def find_mesh_info(self):
        coords = self.mesh.coordinates()
        self.length = len(coords)
        max_v = coords.max(axis=0)
        min_v = coords.min(axis=0)

        self.xmin = min_v[0]
        self.xmax = max_v[0]
        self.ymin = min_v[1]
        self.ymax = max_v[1]

        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin

        self.dim = self.mesh.topology().dim()

        # Collect all vertices that lie on one of the periodic
        # boundaries of the mesh.
        px_mins = []
        px_maxs = []
        py_mins = []
        py_maxs = []
        mesh = self.mesh
        for vertex in df.vertices(mesh):
            if vertex.point().x() == self.xmin:
                px_mins.append(df.Vertex(mesh, vertex.index()))
            elif vertex.point().x() == self.xmax:
                px_maxs.append(df.Vertex(mesh, vertex.index()))

            if vertex.point().y() == self.ymin:
                py_mins.append(df.Vertex(mesh, vertex.index()))
            elif vertex.point().y() == self.ymax:
                py_maxs.append(df.Vertex(mesh, vertex.index()))

        # Collect the indices of vertices on the 'min' boundary
        # and find all vertices on the 'max' boundary which match
        # one of those 'min' vertices.
        indics = []
        indics_pbc = []

        for v1 in px_mins:
            indics.append(v1.index())
            for v2 in px_maxs:
                if v1.point().y() == v2.point().y() and v1.point().z() == v2.point().z() :
                    indics_pbc.append(v2.index())
                    px_maxs.remove(v2)

        for v1 in py_mins:
            indics.append(v1.index())
            for v2 in py_maxs:
                if v1.point().x() == v2.point().x() and v1.point().z() == v2.point().z() :
                    indics_pbc.append(v2.index())
                    py_maxs.remove(v2)
        """
        print self.xmin,self.xmax,self.ymin,self.ymax,self.height,self.width
        print '='*100
        print indics,indics_pbc
        """
        ids = np.array(indics,dtype=np.int32)
        ids_pbc = np.array(indics_pbc,dtype=np.int32)

        #assert len(indics) == len(indics_pbc)

        self.ids = np.array([ids[:], ids[:] + self.length, ids[:] + self.length*2], dtype=np.int32)
        self.ids_pbc = np.array([ids_pbc[:], ids_pbc[:] + self.length,ids_pbc[:] + self.length*2], dtype=np.int32)

        self.ids.shape = (-1,)
        self.ids_pbc.shape = (-1,)


    def modify_m(self,m):
        """
        This method might be not necessary ...
        """
        for i in range(len(self.ids_pbc)):
            j = self.ids_pbc[i]
            k = self.ids[i]
            m[j] = m[k]

    def modify_field(self, v):
        """
        modifiy the corresponding fields, magnetisation m or the volumes of nodes.
        """
        for i in range(len(self.ids_pbc)):
            v[self.ids_pbc[i]] = v[self.ids[i]]



if __name__=="__main__":
    mesh=df.BoxMesh(0,0,0,10,5,1,10,5,1)
    #mesh = df.UnitSquareMesh(2, 2)
    V=df.FunctionSpace(mesh, "Lagrange", 1)
    V3=df.VectorFunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx)
    L = df.assemble(v1 * df.dx)
    print 'before:',K.array()
    pbc=PeriodicBoundary2D(V3)
    print pbc.L.array()
    print 'after',K.array()


