"""
Computation of the demagnetising field using the Fredkin-Koehler
technique and the infamous magpar method.

Rationale: The previous implementation in FemBemFKSolver (child class
of FemBemDeMagSolver) was kind of a mess. This does the same thing in the same
time with less code. Should be more conducive to further optimisation or as
a template for other techniques like the GCR.

"""
import logging
import numpy as np
import dolfin as df
from finmag.native.treecode_bem import compute_solid_angle_single
from finmag.native.treecode_bem import compute_boundary_element
from finmag.native.treecode_bem import build_boundary_matrix

logger = logging.getLogger('finmag')

class MacroGeometry(object):
    def __init__(self, nx=1, ny=1, dx=None, dy=None, Ts=None):
        """
        If Ts is not None the other parameters will be ignored.
        """
        
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.Ts = Ts

        if Ts != None:
            logger.warning("'Ts' is not None, using explicit values in 'Ts'.")
        else:
            if nx < 1 or nx%2==0 or ny<1 or ny%2==0:
                raise Exception('Both nx and ny should larger than 0 and must be odd.')


    def compute_Ts(self, mesh):
        if self.Ts is not None:
            return self.Ts

        dx, dy = self.find_mesh_info(mesh)
        if self.dx is None:
            self.dx = dx
        if self.dy is None:
            self.dy = dy
        
        Ts = []
        for i in range(-self.nx//2+1,self.nx//2+1):
            for j in range(-self.ny//2+1,self.ny//2+1):
                Ts.append([self.dx*i*1.0,self.dy*j*1.0,0])
        
        logger.debug("Creating macro-geometry with demag {} x {} tiles (dxdy: {} x {})".format(self.nx, self.ny, self.dx, self.dy))
           
        self.Ts = Ts
        
        return self.Ts

    def find_mesh_info(self, mesh):
        
        xt = mesh.coordinates()
        
        max_v = xt.max(axis=0)
        min_v = xt.min(axis=0)
        
        sizes = max_v - min_v
        return sizes[0], sizes[1]
        
        

class BMatrixPBC(object):
    
    def __init__(self, mesh, Ts=[(0.,0,0)]):
        self.mesh = mesh
        self.bmesh = df.BoundaryMesh(self.mesh, 'exterior', False)
        self.b2g_map = self.bmesh.entity_map(0).array()
        #self.g2b_map = np.NaN * np.ones(self.mesh.num_vertices(), dtype=int)
        #for (i, val) in enumerate(self.b2g_map):
            #self.g2b_map[val] = i
        self.__compute_bsa()
        self.Ts = np.array(Ts, dtype=np.float)
        
        n = self.bmesh.num_vertices()
        self.bm = np.zeros((n, n))
        self.compute_bmatrix()
    
    def __compute_bsa(self):

        vert_bsa = np.zeros(self.mesh.num_vertices())

        mc = self.mesh.cells()
        xyz = self.mesh.coordinates()
        for i in range(self.mesh.num_cells()):
            for j in range(4):
                tmp_omega = compute_solid_angle_single(
                    xyz[mc[i][j]],
                    xyz[mc[i][(j+1)%4]],
                    xyz[mc[i][(j+2)%4]],
                    xyz[mc[i][(j+3)%4]])

                vert_bsa[mc[i][j]]+=tmp_omega

        vert_bsa = vert_bsa/(4*np.pi) - 1.0

        self.vert_bsa = vert_bsa[self.b2g_map]
    
    def __compute_bmatrix_T(self, T):
        cds = self.bmesh.coordinates()
        face_nodes = np.array(self.bmesh.cells(),dtype=np.int32)
        
        
        be = np.array([0.,0.,0.])
        
        #dof_indices = self.llg.S1.dofmap().dofs()
        #d2v = df.dof_to_vertex_map(self.llg.S1)
        #v2d = df.vertex_to_dof_map(self.llg.S1)
        #vertex_indices_reduced = [d2v[i] for i in dof_indices] 
        for p in range(len(cds)):
            for c in face_nodes:
                i,j,k = c
                compute_boundary_element(cds[p], cds[i], cds[j], cds[k], be, T)
                self.bm[p][i] += be[0]
                self.bm[p][j] += be[1]
                self.bm[p][k] += be[2]
    
    def compute_bmatrix(self):
        
        cds = self.bmesh.coordinates()
        face_nodes = np.array(self.bmesh.cells(),dtype=np.int32)
        
        self.bm[:,:] = 0.0
        
        for T in self.Ts:
            build_boundary_matrix(cds, face_nodes, self.bm, T, len(cds), len(face_nodes))
            
        for p in range(self.bmesh.num_vertices()):
            self.bm[p][p] += self.vert_bsa[p]
            


        
if __name__ == '__main__':
    mesh = df.UnitCubeMesh(1,1,1)
    b = BMatrixPBC(mesh)
    b.compute_bmatrix()
    
    print b.b2g_map, b.g2b_map
    
    bem, b2g_map = compute_bem_fk(df.BoundaryMesh(mesh, 'exterior', False))
    
    #print bem
    #print b2g_map
