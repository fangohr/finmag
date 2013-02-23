from cpython cimport bool
import numpy as np
cimport numpy as np
import time


cdef extern from "common.h":
    ctypedef struct fastsum_plan:
        pass

    fastsum_plan* create_plan()
    void fastsum_finalize(fastsum_plan *plan)
    void update_potential_u1(fastsum_plan *plan,double *u1)
    
    void init_fastsum(fastsum_plan *plan, int N_target, int triangle_num, int p, double mac, int num_limit, double correct_factor)
    void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal, int *triangle_nodes, double *vert_bsa)

    void build_tree(fastsum_plan *plan)
    void bulid_indices_I(fastsum_plan *plan)
    void bulid_indices_II(fastsum_plan *plan)
    
    void fast_sum_I(fastsum_plan *plan, double *phi,double *u1)
    void fast_sum_II(fastsum_plan *plan, double *phi,double *u1)

    void compute_source_nodes_weights(fastsum_plan *plan)
    double solid_angle_single(double *p, double *x1, double *x2, double *x3)
    void boundary_element(double *xp, double *x1, double *x2, double *x3, double *res)
    int get_total_length(fastsum_plan *plan)
    void direct_sum_I(fastsum_plan *plan, double *phi, double *u1)

cdef class FastSum:
    cdef fastsum_plan *_c_plan
    cdef int p
    cdef int num_limit
    cdef double mac
    cdef double correct_factor
    cdef bool type_I
    
    def __cinit__(self,p=3,mac=0.3,num_limit=100,correct_factor=10,type_I=True):
        self.p=p
        self.mac=mac
        self.num_limit=num_limit
        self.correct_factor=correct_factor
        self.type_I=type_I
        assert mac>=0 
        assert p>=1
        assert num_limit>=1
        assert correct_factor>=0
        self._c_plan=create_plan()
        #print 'from cython p=',p,'mac=',mac,'correct_factor=',correct_factor,'num_limit=',num_limit
        if self._c_plan is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self._c_plan is not NULL:
            fastsum_finalize(self._c_plan)
            self._c_plan=NULL


    def init_mesh(self,np.ndarray[double, ndim=2, mode="c"] x_t,
                          np.ndarray[double, ndim=2, mode="c"] t_normal,
                          np.ndarray[int, ndim=2, mode="c"] face_nodes,
                          np.ndarray[double, ndim=1, mode="c"] vert_bsa):

        cdef int num_target,num_faces
        
        num_target = x_t.shape[0]
        num_faces = face_nodes.shape[0]

        init_fastsum(self._c_plan, num_target, num_faces, self.p, self.mac, self.num_limit, self.correct_factor)
        
        init_mesh(self._c_plan,&x_t[0,0],&t_normal[0,0],&face_nodes[0,0],&vert_bsa[0])
        
        compute_source_nodes_weights(self._c_plan)
       
        build_tree(self._c_plan)
        
        
        if self.type_I:
            bulid_indices_I(self._c_plan)
        else:
            bulid_indices_II(self._c_plan)
       


    def fastsum(self,np.ndarray[double, ndim=1, mode="c"] phi,np.ndarray[double, ndim=1, mode="c"] u1):
        update_potential_u1(self._c_plan, &u1[0])
        if self.type_I:
            fast_sum_I(self._c_plan,&phi[0],&u1[0])
        else:
            fast_sum_II(self._c_plan,&phi[0],&u1[0])
            
    def directsum(self,np.ndarray[double, ndim=1, mode="c"] phi,np.ndarray[double, ndim=1, mode="c"] u1):
        update_potential_u1(self._c_plan, &u1[0])
        direct_sum_I(self._c_plan,&phi[0],&u1[0])

    def get_B_length(self):
        return get_total_length(self._c_plan)

    def free_memory(self):
        if self._c_plan is not NULL:
            fastsum_finalize(self._c_plan)
            self._c_plan=NULL


def compute_solid_angle_single(np.ndarray[double, ndim=1, mode="c"] p,
                        np.ndarray[double, ndim=1, mode="c"] x1,
                        np.ndarray[double, ndim=1, mode="c"] x2,
                        np.ndarray[double, ndim=1, mode="c"] x3):

    tmp=solid_angle_single(&p[0], &x1[0], &x2[0], &x3[0])
    return tmp

def compute_boundary_element(np.ndarray[double, ndim=1, mode="c"] xp,
                        np.ndarray[double, ndim=1, mode="c"] x1,
                        np.ndarray[double, ndim=1, mode="c"] x2,
                        np.ndarray[double, ndim=1, mode="c"] x3,
                        np.ndarray[double, ndim=1, mode="c"] res):
    boundary_element(&xp[0], &x1[0], &x2[0], &x3[0], &res[0])
