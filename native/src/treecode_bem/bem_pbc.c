#include "common.h"

void build_matrix_T(double *x_t, int *tri_nodes, double *bm, double *T, int n_node, int n_face) {
	int i,j,k, p, c;
	double be[3];
	double *v, *v1, *v2, *v3;
	
	//#pragma omp parallel for
	#pragma omp parallel for private(i,j,k,p,c,be,v,v1,v2,v3)
	for(p=0; p<n_node; p++){
		
		v = &x_t[3*p];
		
		for(c=0; c<n_face; c++){
			i =  tri_nodes[3*c];
			j =  tri_nodes[3*c+1];
			k =  tri_nodes[3*c+2];
			
			v1 = &x_t[3*i];
			v2 = &x_t[3*j];
			v3 = &x_t[3*k];
			
			boundary_element(v, v1, v2, v3, be, T);
			
			bm[p*n_node+i] += be[0];
			bm[p*n_node+j] += be[1];
			bm[p*n_node+k] += be[2];
		}
		
	}
    
}




