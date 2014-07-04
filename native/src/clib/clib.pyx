from petsc4py.PETSc cimport Vec,  PetscVec

cdef extern from "util.h":
     int norm_c(PetscVec A, PetscVec B)
     int cross_c(PetscVec A, PetscVec B, PetscVec C)

def norm(Vec A, Vec B):
    norm_c(A.vec, B.vec)

def cross(Vec A, Vec B, Vec C):
    """
    Compute C = A x B
    """
    cross_c(A.vec, B.vec, C.vec)
    