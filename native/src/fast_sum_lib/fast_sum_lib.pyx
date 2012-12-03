# cython: profile=True
import numpy as np
cimport numpy as np
import time


cdef extern from "fast_sum.h":
    ctypedef struct fastsum_plan:
        pass

    fastsum_plan* create_plan()
    void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal,\
        int *triangle_nodes, int *tetrahedron_nodes)
    void fastsum_finalize(fastsum_plan *plan)
    void update_charge_density(fastsum_plan *plan,double *m)
    void fastsum_exact(fastsum_plan *plan, double *phi)
    void fastsum(fastsum_plan *plan, double *phi)
    void init_fastsum(fastsum_plan *plan, int N_target, int triangle_p,\
        int tetrahedron_p, int triangle_num, int tetrahedron_num, int p, double mac, int num_limit)

    void build_tree(fastsum_plan *plan)
    void compute_correction(fastsum_plan *plan, double *m, double *phi)
    void compute_source_nodes_weights(fastsum_plan *plan) 

	

cdef class FastSum:
    cdef fastsum_plan *_c_plan

    cdef double mac
    cdef int p
    cdef int num_limit
    cdef triangle_p,tetrahedron_p
    def __cinit__(self,p=4,mac=0.5,num_limit=500,triangle_p=1,tetrahedron_p=0):
        self.num_limit=num_limit
        self.p=p
        self.mac=mac
        self.triangle_p=triangle_p
        self.tetrahedron_p=tetrahedron_p
        self._c_plan=create_plan()
        print 'from cython p=',p,'mac=',mac,'triangle_p=',triangle_p,'vn=',tetrahedron_p,'num_limit=',num_limit
        if self._c_plan is NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        if self._c_plan is not NULL:
            fastsum_finalize(self._c_plan)
            self._c_plan=NULL
            

    def init_mesh(self,np.ndarray[double, ndim=2, mode="c"] x_t,
                          np.ndarray[double, ndim=2, mode="c"] t_normal,
                          np.ndarray[int, ndim=2, mode="c"] face_nodes,
                          np.ndarray[int, ndim=2, mode="c"] tet_nodes  ):
        cdef int N,M,num_faces
        
        M=x_t.shape[0]
        num_faces=face_nodes.shape[0]
        num_tet=tet_nodes.shape[0]
        init_fastsum(self._c_plan,M,self.triangle_p,self.tetrahedron_p,num_faces,num_tet,self.p,self.mac,self.num_limit)
         
        init_mesh(self._c_plan,&x_t[0,0],&t_normal[0,0],&face_nodes[0,0],&tet_nodes[0,0])
        compute_source_nodes_weights(self._c_plan)
        build_tree(self._c_plan)

    def update_charge(self,np.ndarray[double, ndim=1, mode="c"] m):
        update_charge_density(self._c_plan, &m[0])
        #print 'update charge ok'

    def exactsum(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum_exact(self._c_plan,&phi[0])
        
    def fastsum(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum(self._c_plan,&phi[0])

    def compute_correction(self,np.ndarray[double, ndim=1, mode="c"] m,np.ndarray[double, ndim=1, mode="c"] phi):
        compute_correction(self._c_plan,&m[0],&phi[0])
    
    def free_memory(self):
        if self._c_plan is not NULL:
            fastsum_finalize(self._c_plan)
            self._c_plan=NULL
        