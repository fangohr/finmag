# cython: profile=True
import numpy as np
cimport numpy as np
import time


cdef extern from "fast_sum.h":
    ctypedef struct fastsum_plan:
        pass

    fastsum_plan* create_plan()
    void init_mesh(fastsum_plan *plan, double *x_s, double *x_t, double *t_normal, int *face_nodes)
    void fastsum_finalize(fastsum_plan *plan)
    void update_charge_density(fastsum_plan *plan, double *m,double *weight)
    void fastsum_exact(fastsum_plan *plan, double *phi)
    void fastsum(fastsum_plan *plan, double *phi)
    void init_fastsum(fastsum_plan *plan, int N_source, int N_target, \
                int surface_n, int volume_n, int num_faces, int p, double mac, int num_limit) 
    void build_tree(fastsum_plan *plan)
    void compute_correction(fastsum_plan *plan, double *m, double *phi)
    void update_charge_directly(fastsum_plan *plan, double *weight)

	

cdef class FastSum:
    cdef fastsum_plan *_c_plan

    cdef double mac
    cdef int p
    cdef int num_limit
    cdef surface_n,volume_n
    def __cinit__(self,p=4,mac=0.5,num_limit=500,surface_n=3,volume_n=2):
        self.num_limit=num_limit
        self.p=p
        self.mac=mac
        self.surface_n=surface_n
        self.volume_n=volume_n
        self._c_plan=create_plan()
        print 'from cython p=',p,'mac=',mac,'surface_n=',surface_n,'num_limit=',num_limit
        if self._c_plan is NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        if self._c_plan is NULL:
            fastsum_finalize(self._c_plan)

    def init_mesh(self,np.ndarray[double, ndim=2, mode="c"] x_s,
                         np.ndarray[double, ndim=2, mode="c"] x_t,
                          np.ndarray[double, ndim=2, mode="c"] t_normal,
                          np.ndarray[int, ndim=2, mode="c"] face_nodes):
        cdef int N,M,num_faces
        
        N,M=x_s.shape[0],x_t.shape[0]
        num_faces=face_nodes.shape[0]
        init_fastsum(self._c_plan,N,M,self.surface_n,self.volume_n,num_faces,self.p,self.mac,self.num_limit)
         
        init_mesh(self._c_plan,&x_s[0,0],&x_t[0,0],&t_normal[0,0],&face_nodes[0,0])
        build_tree(self._c_plan)
        

    def update_charge(self,np.ndarray[double, ndim=1, mode="c"] m,np.ndarray[double, ndim=1, mode="c"] weight):
        update_charge_density(self._c_plan, &m[0],&weight[0])

    def update_charge_directly(self,np.ndarray[double, ndim=1, mode="c"] weight):
        update_charge_directly(self._c_plan, &weight[0])         

    def exactsum(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum_exact(self._c_plan,&phi[0])
        

    def fastsum(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum(self._c_plan,&phi[0])

    def compute_correction(self,np.ndarray[double, ndim=1, mode="c"] m,np.ndarray[double, ndim=1, mode="c"] phi):
        compute_correction(self._c_plan,&m[0],&phi[0])
        