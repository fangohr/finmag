# cython: profile=True
import numpy as np
cimport numpy as np
import time


cdef extern from "fast_sum.h":
    ctypedef struct fastsum_plan:
        pass

    fastsum_plan* create_plan()
    void fastsum_finalize(fastsum_plan *plan)
    void update_potential_u1(fastsum_plan *plan,double *u1)
    void fastsum(fastsum_plan *plan, double *phi,double *u1)

    void init_fastsum(fastsum_plan *plan, int N_target, int triangle_p,int triangle_num, int p, double mac, int num_limit)
    void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal, int *triangle_nodes, int *g2b, double *vert_bsa)

    void build_tree(fastsum_plan *plan)
    void bulid_indices(fastsum_plan *plan)
    #void compute_correction(fastsum_plan *plan, double *m, double *phi)
    void compute_source_nodes_weights(fastsum_plan *plan)
    void compute_triangle_source_nodes(fastsum_plan *plan) 
    double solid_angle_single(double *p, double *x1, double *x2, double *x3)
    void boundary_element(double *xp, double *x1, double *x2, double *x3, double *res)
    void copy_B(fastsum_plan *plan, double *B, int n)


cdef class FastSum:
    cdef fastsum_plan *_c_plan

    cdef double mac
    cdef int p
    cdef int num_limit
    cdef int triangle_p
    def __cinit__(self,triangle_p=1,p=3,mac=0.3,num_limit=100):
        self.triangle_p=triangle_p
        self.num_limit=num_limit
        self.p=p
        self.mac=mac
        assert mac>=0 and mac<=1
        assert triangle_p>=0 and triangle_p<=3
        assert p>=1
        assert num_limit>=1

        self._c_plan=create_plan()
        print 'from cython p=',p,'mac=',mac,'triangle_p=',triangle_p,'num_limit=',num_limit
        if self._c_plan is NULL:
            raise MemoryError()



    def __dealloc__(self):
        if self._c_plan is not NULL:
            fastsum_finalize(self._c_plan)
            self._c_plan=NULL


    def init_mesh(self,np.ndarray[double, ndim=2, mode="c"] x_t,
                          np.ndarray[double, ndim=2, mode="c"] t_normal,
                          np.ndarray[int, ndim=2, mode="c"] face_nodes,
                          np.ndarray[int, ndim=1, mode="c"] g2b,
                          np.ndarray[double, ndim=1, mode="c"] vert_bsa):

        cdef int N,M,num_faces
        M=x_t.shape[0]
        num_faces=face_nodes.shape[0]

        init_fastsum(self._c_plan,M,self.triangle_p,num_faces,self.p,self.mac,self.num_limit)
        #print 'init sum okay'
        init_mesh(self._c_plan,&x_t[0,0],&t_normal[0,0],&face_nodes[0,0],&g2b[0],&vert_bsa[0])
        #print 'init mesh okay'
        compute_triangle_source_nodes(self._c_plan)
        #print 'compute triangle nodes okay'
        build_tree(self._c_plan)
        #print 'init okay from cython'
        bulid_indices(self._c_plan)
        #print 'build okay from cython'
        compute_source_nodes_weights(self._c_plan)
        #print 'init mesh okay from cython'
        

    def fastsum(self,np.ndarray[double, ndim=1, mode="c"] phi,np.ndarray[double, ndim=1, mode="c"] u1):
        update_potential_u1(self._c_plan, &u1[0])
        fastsum(self._c_plan,&phi[0],&u1[0])


    def compute_B(self,np.ndarray[double, ndim=1, mode="c"] B):
        n=len(B)
        copy_B(self._c_plan, &B[0],n)


    def free_memory(self):
        if self._c_plan is not NULL:
            fastsum_finalize(self._c_plan)
            self._c_plan=NULL


def compute_solid_angle_single(np.ndarray[double, ndim=1, mode="c"] p,
                        np.ndarray[double, ndim=1, mode="c"] x1,
                        np.ndarray[double, ndim=1, mode="c"] x2,
                        np.ndarray[double, ndim=1, mode="c"] x3):
    #cdef double tmp
    tmp=solid_angle_single(&p[0], &x1[0], &x2[0], &x3[0])
    return tmp

def compute_boundary_element(np.ndarray[double, ndim=1, mode="c"] xp,
                        np.ndarray[double, ndim=1, mode="c"] x1,
                        np.ndarray[double, ndim=1, mode="c"] x2,
                        np.ndarray[double, ndim=1, mode="c"] x3,
                        np.ndarray[double, ndim=1, mode="c"] res):
    boundary_element(&xp[0], &x1[0], &x2[0], &x3[0], &res[0])