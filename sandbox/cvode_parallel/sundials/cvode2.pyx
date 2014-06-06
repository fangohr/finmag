from libc.string cimport memcpy

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np
from petsc4py import PETSc
from petsc4py.PETSc cimport Vec,  PetscVec
#from petsc4py.PETSc cimport Comm as MPI_Comm

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

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

cdef inline  copy_arr2nv(np.ndarray[realtype, ndim=1,mode='c'] np_x, N_Vector v):
    cdef long int n = (<N_VectorContent_Parallel>v.content).local_length
    cdef void* data_ptr=<void *>np_x.data
    memcpy((<N_VectorContent_Parallel>v.content).data, data_ptr, n*sizeof(double))
    
    return 0
    
cdef inline copy_nv2arr(N_Vector v, np.ndarray[realtype, ndim=1, mode='c'] np_x):
    cdef long int n = (<N_VectorContent_Parallel>v.content).local_length
    cdef double* v_data = (<N_VectorContent_Parallel>v.content).data
    
    memcpy(np_x.data, v_data, n*sizeof(realtype))
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
    
    cdef public double t
    cdef public np.ndarray y
    cdef double rtol, atol
    cdef Vec _spin
    cdef Vec y_dot  # time derivative of y
    cdef N_Vector y_nv  # the N_Vector version of y
    
    cdef void *cvode_mem
    cdef void *rhs_fun
    cdef callback_fun
    cdef cv_userdata user_data
    cdef int MODIFIED_GS

    
    cdef long int nsteps,nfevals,njevals

    def set_options(self, rtol, atol, max_num_steps):
        self.rtol = rtol
        self.atol = atol
        self.max_num_steps = max_num_steps

        # Set tolerances
        flag = CVodeSStolerances(self.cvode_mem, self.rtol, self.atol)
        
        # Set maximum number of iteration steps (?)
        flag = CVodeSetMaxNumSteps(self.cvode_mem, mxsteps)

        # Set options for the CVODE scaled, preconditioned GMRES linear solver, CVSPGMR
        flag = CVSpgmr(self.cvode_mem, PREC_NONE, 300);
        #flag = CVSpilsSetGSType(self.cvode_mem, 1);
    
    def __cinit__(self, Vec spin, callback_fun, rtol=1e-8, atol=1e-8, max_num_steps=100000):

        self.t = 0
        self._spin = spin
        self.y_dot = spin.duplicate()

        self.callback_fun = callback_fun
        
        cdef np.ndarray[double, ndim=1, mode="c"] y = np.zeros(spin.getLocalSize())
        
        self.y = y
        
        #VecGetArray(self._spin.vec,&y[0])
        #self.y_nv = N_VNew_Serial(self.y.getLocalSize())
        
        cdef MPI_Comm comm_c = PETSC_COMM_WORLD
        
        #self.y_nv = N_VNew_Parallel(comm_c, spin.getLocalSize(), spin.getSize())
        
        #VecGetArray(spin.vec,&y[0])
        #how can we use the orginal petsc array rather than the numpy array?
        #self.y_nv = N_VMake_Serial(spin.getLocalSize(),&y[0])
        self.y_nv = N_VMake_Parallel(comm_c, spin.getLocalSize(), spin.getSize(), &y[0])
        
        #VecRestoreArray(spin.vec,&y[0])
        
        self.rhs_fun = <void *>cv_rhs

        self.user_data = cv_userdata(<void*>self.callback_fun,
                                     <void *>self._spin,<void *>self.y_dot)

        self.MODIFIED_GS = 1
        
        self.init_ode()

        self.set_options(rtol, atol, max_num_steps)
        

    def init_ode(self):
        self.cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
        
        #self.cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);


    def set_initial_value(self,np.ndarray[double, ndim=1, mode="c"] spin, t):
        self.t = t
        #
        copy_arr2nv(spin, self.y_nv)

        flag = CVodeSetUserData(self.cvode_mem, <void*>&self.user_data);
        self.check_flag(flag,"CVodeSetUserData")

        flag = CVodeInit(self.cvode_mem, <CVRhsFn>self.rhs_fun, t, self.y_nv)
        self.check_flag(flag,"CVodeInit")


    cpdef int run_until(self, double tf) except -1:
        cdef int flag
        cdef double tret
        
        flag = CVodeStep(self.cvode_mem, tf, self.y_nv, &tret, CV_NORMAL)
        self.check_flag(flag,"CVodeStep")
        
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
    
    def test_petsc(self, Vec x):
        cdef PetscInt size
        VecGetLocalSize(x.vec, &size)
        print "length from cython = %d"%size
        comm = PETSc.COMM_WORLD
        print 'from cython, common:', comm, comm.Get_rank(),comm.Get_size()

    def __repr__(self):
        s = []
        s.append("nsteps = %d,"      % self.nsteps)
        s.append("nfevals = %d,"     % self.nfevals)
        s.append("njevals = %d.\n"     % self.njevals)

        return "(%s)" % ("\n".join(s))
    
    def __str__(self):
        return '%s%s' % (self.__class__.__name__, self.__repr__())

