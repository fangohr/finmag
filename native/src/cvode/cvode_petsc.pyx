from libc.string cimport memcpy

# Import the C-level symbols of numpy
cimport numpy as np_c
from petsc4py import PETSc
from petsc4py.PETSc cimport Vec,  PetscVec

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np_c.import_array()

cdef extern from "petsc/petscvec.h":
     PetscErrorCode VecGetLocalSize(PetscVec,PetscInt*) #for testing
     PetscErrorCode VecGetArray(PetscVec, PetscReal *x) #check the type later, petscreal or petscsalar?
     PetscErrorCode VecRestoreArray(PetscVec, PetscReal *x)
     PetscErrorCode VecPlaceArray(PetscVec, PetscReal *x)
     PetscErrorCode VecResetArray(PetscVec)
     PetscErrorCode VecCreateMPIWithArray(MPI_Comm comm, PetscInt bs,PetscInt n,PetscInt N,const PetscScalar array[],PetscVec)
     
cdef struct cv_userdata:
    void *rhs_fun
    void *y
    void *y_dot

cdef inline  copy_arr2nv(np_c.ndarray[realtype, ndim=1,mode='c'] np_x, N_Vector v):
    cdef long int n = (<N_VectorContent_Parallel>v.content).local_length
    cdef void* data_ptr=<void *>np_x.data
    memcpy((<N_VectorContent_Parallel>v.content).data, data_ptr, n*sizeof(double))
    
    return 0
    
cdef inline copy_nv2arr(N_Vector v, np_c.ndarray[realtype, ndim=1, mode='c'] np_x):
    cdef long int n = (<N_VectorContent_Parallel>v.content).local_length
    cdef double* v_data = (<N_VectorContent_Parallel>v.content).data
    
    memcpy(np_x.data, v_data, n*sizeof(realtype))
    return 0

cdef int jtv(N_Vector v, N_Vector Jv, realtype t, N_Vector y, N_Vector fy, void *user_data, N_Vector tmp) except -1:
     
     return 0

cdef int cv_rhs(realtype t, N_Vector yv, N_Vector yvdot, void* user_data) except -1:

    cdef cv_userdata *ud = <cv_userdata *>user_data

    #cdef int size = (<N_VectorContent_Serial>yvdot.content).length
    #cdef double *y = (<N_VectorContent_Serial>yv.content).data
    
    cdef Vec y = <Vec>ud.y
    cdef Vec ydot = <Vec>ud.y_dot
    
    VecPlaceArray(y.vec, (<N_VectorContent_Parallel>yv.content).data)
    VecPlaceArray(ydot.vec, (<N_VectorContent_Parallel>yvdot.content).data)
    
    (<object>ud.rhs_fun)(t,y,ydot)
    
    VecResetArray(y.vec)
    VecResetArray(ydot.vec)

    return 0

cdef class CvodeSolver(object):
    
    cdef public double cur_t
    cdef public np_c.ndarray y_np
    cdef double rtol, atol
    cdef int max_num_steps
    cdef Vec y
    cdef Vec y_dot  # time derivative of y
    cdef N_Vector y_nv  # the N_Vector version of y
    
    cdef void *cvode_mem
    cdef void *cv_rhs
    cdef callback_fun
    cdef cv_userdata user_data
    cdef jac_fun    

    cdef long int nsteps,nfevals,njevals

    def __cinit__(self, callback_fun, t0, y0, jac_fun=None, rtol=1e-8, atol=1e-8, max_num_steps=100000):
        # y0 should be a petsc array, and we update this y0 automatically when call the given call_back function
	
        # Create the CVODE memory block and to specify the solution method (linear multistep method and nonlinear solver iteration type)
        self.cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
        #self.cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
	
	self.jac_fun = jac_fun
        self.init_ode(callback_fun, t0, y0)
        self.set_options(rtol, atol, max_num_steps)

    def init_ode(self, callback_fun, t0, y0):
        """
        
        """
        self.callback_fun = callback_fun
        self.cv_rhs = <void *>cv_rhs  # wrapper for callback_fun (which is a Python function)

        self.y = y0
        self.y_dot = self.y.duplicate()
        self.cur_t = t0

        self.user_data = cv_userdata(<void*>self.callback_fun,
                                     <void *>self.y,<void *>self.y_dot)

        cdef MPI_Comm comm_c = PETSC_COMM_WORLD
        cdef np_c.ndarray[double, ndim=1, mode="c"] y_np = y0.getArray()
        self.y_np = y_np
        self.y_nv = N_VMake_Parallel(comm_c, y0.getLocalSize(), y0.getSize(), &y_np[0])
        
        flag = CVodeInit(self.cvode_mem, <CVRhsFn>self.cv_rhs, t0, self.y_nv)
        self.check_flag(flag,"CVodeInit")

        flag = CVodeSetUserData(self.cvode_mem, <void*>&self.user_data);
        self.check_flag(flag,"CVodeSetUserData")

    def set_options(self, rtol, atol, max_num_steps=100000):
        self.rtol = rtol
        self.atol = atol
        self.max_num_steps = max_num_steps

        # Set tolerances
        flag = CVodeSStolerances(self.cvode_mem, self.rtol, self.atol)
        
        # Set maximum number of iteration steps (?)
        flag = CVodeSetMaxNumSteps(self.cvode_mem, max_num_steps)

        # Set options for the CVODE scaled, preconditioned GMRES linear solver, CVSPGMR
        flag = CVSpgmr(self.cvode_mem, PREC_NONE, 300);
        #flag = CVSpilsSetGSType(self.cvode_mem, 1);

    #def set_initial_value(self,np.ndarray[double, ndim=1, mode="c"] spin, t):
    #    self.t = t
    #    #
    #    copy_arr2nv(spin, self.y_nv)

    cpdef int run_until(self, double tf) except -1:
        cdef int flag
        cdef double tret
        
        flag = CVodeStep(self.cvode_mem, tf, self.y_nv, &tret, CV_NORMAL)
        self.check_flag(flag,"CVodeStep")
        self.cur_t = tf
        
        return 0

    cpdef int advance_time(self, double tf) except -1:
        
        self.run_until(tf)
        
        return 0


    def check_flag(self, flag, fun_name):
        if flag<0:
            raise Exception("Run %s failed!"%fun_name)

    def stat(self):
        
        CVodeGetNumSteps(self.cvode_mem, &self.nsteps);
        CVodeGetNumRhsEvals(self.cvode_mem, &self.nfevals);
        #CVDlsGetNumJacEvals(self.cvode_mem, &self.njevals)
        
        return self.__str__()
    
    def get_current_step(self):
        cdef double step
        CVodeGetCurrentStep(self.cvode_mem, &step)
        return step

    def __repr__(self):
        s = []
        s.append("nsteps = %d,"      % self.nsteps)
        s.append("nfevals = %d,"     % self.nfevals)
        s.append("njevals = %d.\n"     % self.njevals)

        return "(%s)" % ("\n".join(s))
    
    def __str__(self):
        return '%s%s' % (self.__class__.__name__, self.__repr__())

