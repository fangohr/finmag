#include "util.h"

int norm_c(Vec A, Vec B){
  PetscReal *a, *b;
  int i,j, na=0;
  
  VecGetArray(A, &a);
  VecGetArray(B, &b);

  VecGetLocalSize(A,&na);
  
  for (i = 0; i < na; i += 3) {
    j = i/3;
    b[j] = sqrt(a[i]*a[i]+a[i+1]*a[i+1]+a[i+2]*a[i+2]);
  }
  
  VecRestoreArray(A, &a);
  VecRestoreArray(B, &b);
  
  return 0;
}

int cross_c(Vec A, Vec B, Vec C) {
	
    PetscReal *a, *b, *c;

    int i,nc=0;
    
    VecGetArray(A, &a);
    VecGetArray(B, &b);
    VecGetArray(C, &c);
    
    VecGetLocalSize(C,&nc);
    
    //assert(na==nb);
    //assert(nb==nc);

    for (i = 0; i < nc; i += 3) {
      c[i] = a[i+1]*b[i+2] - a[i+2]*b[i+1];
      c[i+1] = a[i+2]*b[i] - a[i]*b[i + 2];
      c[i+2] = a[i]*b[i+1] - a[i+1]*b[i];
    }

    VecRestoreArray(A, &a);
    VecRestoreArray(B, &b);
    VecRestoreArray(C, &c);
    
    return 0;
}

