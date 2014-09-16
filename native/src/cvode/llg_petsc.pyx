from petsc4py.PETSc cimport Vec,  PetscVec

cdef extern from "llg.h":
     int llg_rhs(PetscVec M, PetscVec H, PetscVec dM_dt, PetscVec alpha_v,  double gamma, int do_precession, double char_freq) 
    
def compute_dm_dt(Vec M, Vec H, Vec dM_dt, Vec alpha_v, gamma, do_precession, char_freq):
    llg_rhs(M.vec, H.vec, dM_dt.vec, alpha_v.vec, gamma, do_precession, char_freq)
    