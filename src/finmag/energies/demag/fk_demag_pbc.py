"""
Computation of the demagnetising field using the Fredkin-Koehler
technique and the infamous magpar method.

Rationale: The previous implementation in FemBemFKSolver (child class
of FemBemDeMagSolver) was kind of a mess. This does the same thing in the same
time with less code. Should be more conducive to further optimisation or as
a template for other techniques like the GCR.

"""
import numpy as np
import dolfin as df

from finmag.native.llg import compute_bem_fk
from finmag.native.treecode_bem import compute_solid_angle_single
from finmag.native.treecode_bem import compute_boundary_element

class BMatrixPBC(object):
    
    def __init__(self, S3, Ts=[(0.,0,0)]):
        self.mesh = S3.mesh()
        self.bmesh = df.BoundaryMesh(self.mesh, 'exterior', False)
        self.b2g_map = self.bmesh.entity_map(0).array()
        self.__compute_bsa()
        self.Ts = np.array(Ts)
        self.S3 = S3
        
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

        vert_bsa = vert_bsa/(4*np.pi)

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
                compute_boundary_element(cds[p], cds[i]+T, cds[j]+T, cds[k]+T, be)
                self.bm[p][i] += be[0]
                self.bm[p][j] += be[1]
                self.bm[p][k] += be[2]
    
    def compute_bmatrix(self):
        
        self.bm[:,:] = 0.0

        
        for T in self.Ts:
            print T
            self.__compute_bmatrix_T(T)

        for p in range(self.bmesh.num_vertices()):
            self.bm[p][p] += self.vert_bsa[p] - 1


        #bbt = df.BoundaryBoxTree()
        #bbt.build(self.bmesh)
        #bmesh_coords = self.bmesh.coordinates()
        #def find_closest_mesh_point(idx):
        #    pt = bmesh_coords[idx]
        #    cell_idx, dist = bbt.compute_closest_entity(df.Point(pt[0], pt[1], pt[2]))
        dofmap = self.S3.dofmap()
        S1 = df.FunctionSpace(self.mesh, "Lagrange", 1, constrained_domain=dofmap.constrained_domain)
        
        d2v = df.dof_to_vertex_map(S1)
        v2d = df.vertex_to_dof_map(S1)
        
        coords = self.bmesh.coordinates()
        maxdist = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
        for T in self.Ts:
            if np.linalg.norm(T) == 0 or np.linalg.norm(T) > maxdist:
                continue
            for p in range(self.bmesh.num_vertices()):
                p2 = d2v[v2d[p]]
                if np.allclose(coords[p2], coords[p] - T):
                    print "[DDD] Adding correction.", T, p, p2, self.vert_bsa[p], self.vert_bsa[p2]
                    self.bm[p][p] += self.vert_bsa[p2]         


        
if __name__ == '__main__':
    mesh = df.UnitCubeMesh(1,1,1)
    b = BMatrixPBC(mesh)
    b.compute_bmatrix()
    
    print b.b2g_map
    
    bem, b2g_map = compute_bem_fk(df.BoundaryMesh(mesh, 'exterior', False))
    
    #print bem
    #print b2g_map