
import numpy as np
cimport numpy as np


cdef extern from "demag_nfft.h":
    ctypedef struct fastsum_plan:
        pass
    void fastsum_init_guru(fastsum_plan *plan, int N_total, int M_total, int nn, int m, int p, double eps_I, double eps_B)
    void fastsum_trafo(fastsum_plan *plan)
    void fastsum_precompute(fastsum_plan *plan)
    fastsum_plan* create_plan()
    void init_mesh(fastsum_plan *plan,double *x_s,double *x_t)
    void fastsum_finalize(fastsum_plan *plan)
    void update_charge(fastsum_plan *plan, double *charge)
    void get_phi(fastsum_plan *plan, double *phi)
    void fastsum_exact(fastsum_plan *plan)
    void fastsum_finalize(fastsum_plan *plan) 


cdef class FastSum:
    cdef fastsum_plan *_c_plan
    cdef int n,m,p
    cdef eps_I,eps_B
    def __cinit__(self,n=128,m=4,p=3):
        self.n=n
        self.m=m
        self.p=p
        self.eps_I = 1.0 * p / n
        self.eps_B = 0.0
        self._c_plan=create_plan()
        if self._c_plan is NULL:
            raise MemoryError()
    
    def __dealloc__(self):
        if self._c_plan is NULL:
            fastsum_finalize(self._c_plan)

    def init_mesh(self,np.ndarray[double, ndim=2, mode="c"] x_s,np.ndarray[double, ndim=2, mode="c"] x_t):
        cdef int N,M

        N,M=x_s.shape[0],x_t.shape[0]
        
        fastsum_init_guru(self._c_plan,N,M,self.n,self.m,self.p,self.eps_I,self.eps_B)
        #print 'init_guru okay from FastSum'
        init_mesh(self._c_plan,&x_s[0,0],&x_t[0,0])
        #print 'init_mesh okay from FastSum'
        fastsum_precompute(self._c_plan)
        

  
    def sum_exact(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum_exact(self._c_plan)
        get_phi(self._c_plan,&phi[0])
        

    def update_charge(self,np.ndarray[double, ndim=1, mode="c"] cf):
        update_charge(self._c_plan, &cf[0])
        #print 'update charge okay from FastSum'
        
        
    def compute_phi(self,np.ndarray[double, ndim=1, mode="c"] phi):
        fastsum_trafo(self._c_plan)
        get_phi(self._c_plan,&phi[0])