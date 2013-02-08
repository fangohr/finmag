import os
import time
import dolfin as df
import numpy as np
import finmag


class PeriodicBoundaryX(df.SubDomain):
    
    def __init__(self,xmin,width,dim):
        super(PeriodicBoundaryX, self).__init__()
        self.xmin=xmin
        self.width=width
        self.dim=dim
        

    def inside(self, x, on_boundary):
        return df.near(x[0],self.xmin) and on_boundary

    def map(self, x, y):
        y[0] = x[0] - self.width
        y[1] = x[1]
        if self.dim==3:
            y[2] = x[2]

class PeriodicBoundaryY(df.SubDomain):
    
    def __init__(self,ymin,height,dim):
        super(PeriodicBoundaryY, self).__init__()
        self.ymin=ymin
        self.height=height
        self.dim=dim

    def inside(self, x, on_boundary):
        return df.near(x[1],self.ymin) and on_boundary

    def map(self, x, y):
        y[0] = x[0] 
        y[1] = x[1] - self.height
        if self.dim==3:
            y[2] = x[2]            
            

class PeriodicBoundary2D(object):
    """
    Periodic Boundary condition in xy-plane potentially can be applied to the energy_base class.
    """
    def __init__(self,V,K,unit_length=1.0):
        self.V=V
        self.K=K
        self.unit_length=unit_length
        self.mesh=V.mesh()
        
        self.find_mesh_info()
        
        v = df.TestFunction(V)
        if isinstance(V, df.VectorFunctionSpace):
            self.L=df.assemble(df.dot(v, df.Constant((1, 1, 1))) * df.dx)
        else: 
            self.L=df.assemble(v * df.dx)
            
        self.apply_pbc()
        
        
    def find_mesh_info(self):
        xt=self.mesh.coordinates()
        max_v=xt.max(axis=0)
        min_v=xt.min(axis=0)
        
        self.xmin=min_v[0]
        self.xmax=max_v[0]
        self.ymin=min_v[1]
        self.ymax=max_v[1]
        
        self.width=self.xmax-self.xmin
        self.height=self.ymax-self.ymin
        
        self.dim=self.mesh.topology().dim()
        
        px_mins=[]
        px_maxs=[]
        py_mins=[]
        py_maxs=[]
        mesh=self.mesh
        for vertex in df.vertices(mesh):
            if vertex.point().x()==self.xmin:
                px_mins.append(df.Vertex(mesh,vertex.index()))
            elif vertex.point().x()==self.xmax:
                px_maxs.append(df.Vertex(mesh,vertex.index()))
            
            if vertex.point().y()==self.ymin:
                py_mins.append(df.Vertex(mesh,vertex.index()))
            elif vertex.point().y()==self.ymax:
                py_maxs.append(df.Vertex(mesh,vertex.index()))
                
        indics=[]
        indics_pbc=[]

        for v1 in px_mins:
            indics.append(v1.index())
            for v2 in px_maxs:
                if v1.point().y()==v2.point().y() and v1.point().z()==v2.point().z() :
                    indics_pbc.append(v2.index())
                    px_maxs.remove(v2)

        for v1 in py_mins:
            indics.append(v1.index())
            for v2 in py_maxs:
                if v1.point().x()==v2.point().x() and v1.point().z()==v2.point().z() :
                    indics_pbc.append(v2.index())
                    py_maxs.remove(v2)
        """
        print self.xmin,self.xmax,self.ymin,self.ymax,self.height,self.width
        print '='*100
        print indics,indics_pbc
        """
        self.ids=indics
        self.ids_pbc=indics_pbc
        
    
    def apply_pbc(self):
        bcx=df.PeriodicBC(self.V,PeriodicBoundaryX(self.xmin,self.width,self.dim))
        bcy=df.PeriodicBC(self.V,PeriodicBoundaryY(self.ymin,self.height,self.dim))
        bcx.apply(self.K,self.L)
        bcy.apply(self.K,self.L)
        
        """
        must call the sequence in order
        also must import finmag at the beginning of the file (about order stuff?)
        """
        for i in range(len(self.ids_pbc)):
            self.L[self.ids_pbc[i]]=self.L[self.ids[i]]
        


if __name__=="__main__":
    mesh=df.BoxMesh(0,0,0,10,5,1,10,5,1)
    #mesh = df.UnitSquareMesh(2, 2)  
    V=df.FunctionSpace(mesh, "Lagrange", 1)
    u1 = df.TrialFunction(V)
    v1 = df.TestFunction(V)
    K = df.assemble(df.inner(df.grad(u1), df.grad(v1)) * df.dx)
    L = df.assemble(v1 * df.dx)
    print 'before:',K.array()
    pbc=PeriodicBoundary2D(V,K)
    print pbc.L.array()
    print 'after',K.array()
    
    