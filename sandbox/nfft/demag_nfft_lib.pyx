
import numpy as np
cimport numpy as np
import time


cdef extern from "fast_sum.h":
    ctypedef struct fastsum_plan:
        pass

    fastsum_plan* create_plan()
    void init_mesh(fastsum_plan *plan,double *x_s,double *x_t)
    void fastsum_finalize(fastsum_plan *plan)
    void update_charge_density(fastsum_plan *plan, double *density)
    void fastsum_exact(fastsum_plan *plan, double *phi)
    void fastsum(fastsum_plan *plan, double *phi)
    void init_fastsum(fastsum_plan *plan, int N_source, int N_target, int p,double mac,int num_limit) 
    void build_tree(fastsum_plan *plan)
	

cdef class FastSum:
    cdef fastsum_plan *_c_plan

    cdef double mac
    cdef int p
    cdef int num_limit
    def __cinit__(self,p=4,mac=0.5,num_limit=500):
        self.num_limit=num_limit
        self.p=p
        self.mac=mac
        self._c_plan=create_plan()
        if self._c_plan is NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        if self._c_plan is NULL:
            fastsum_finalize(self._c_plan)

    def init_mesh(self,np.ndarray[double, ndim=2, mode="c"] x_s,np.ndarray[double, ndim=2, mode="c"] x_t):
        cdef int N,M
        print 'length=',N,M
        
        N,M=x_s.shape[0],x_t.shape[0]
        init_fastsum(self._c_plan,N,M,self.p,self.mac,self.num_limit)
         
        init_mesh(self._c_plan,&x_s[0,0],&x_t[0,0])
        build_tree(self._c_plan)
        

    def update_charge(self,np.ndarray[double, ndim=1, mode="c"] cf):
        update_charge_density(self._c_plan, &cf[0])        

    def exactsum(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum_exact(self._c_plan,&phi[0])
        

    def fastsum(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum(self._c_plan,&phi[0])
        