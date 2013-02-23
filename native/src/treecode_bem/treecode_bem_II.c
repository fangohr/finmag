#include "common.h"


void bulid_indices_single_II(fastsum_plan *plan, struct octree_node *tree,
        int index, int *in, double *value, int compute_bm) {

    int i, j;
    double r,dx, dy, dz;
    double *p0, *p1, *p2, *p3;
    double omega[3];
    int k1, k2, k3;

    r = pow2(plan->x_t[3 * index] - tree->x)
            + pow2(plan->x_t[3 * index + 1] - tree->y)
            + pow2(plan->x_t[3 * index + 2] - tree->z);

    r = sqrt(r);

    if (r > tree->radius + plan->r_eps) {
        return;
    }


    if (tree->num_children > 0) {

        for (i = 0; i < tree->num_children; i++) {
            bulid_indices_single_II(plan, tree->children[i], index, in, value, compute_bm);
        }
        return;

    } else {


        for (i = tree->begin; i < tree->end; i++) {

            dx = plan->x_s[3 * i] - plan->x_t[3 * index];
            dy = plan->x_s[3 * i + 1] - plan->x_t[3 * index + 1];
            dz = plan->x_s[3 * i + 2] - plan->x_t[3 * index + 2];

            r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r <= plan->r_eps) {

                j = plan->x_s_ids[i];
                p0 = &plan->x_t[3 * index];

                k1 = plan->triangle_nodes[3 * j];
                p1 = &plan->x_t[3 * k1];
                in[k1] = 1;

                k2 = plan->triangle_nodes[3 * j + 1];
                p2 = &plan->x_t[3 * k2];
                in[k2] = 1;

                k3 = plan->triangle_nodes[3 * j + 2];
                p3 = &plan->x_t[3 * k3];
                in[k3] = 1;

                if (compute_bm > 0) {
                    boundary_element(p0, p1, p2, p3, omega);
                    value[k1] += omega[0];
                    value[k2] += omega[1];
                    value[k3] += omega[2];
                }
            }
        }
        return;
    }

}

void bulid_indices_II(fastsum_plan *plan) {

    int i, j;

    int *indices_n = malloc(plan->N_target * sizeof ( int));
    double *values = malloc(plan->N_target * sizeof ( double));


    int tmp_length_n = 0;
    int total_length_n = 0;

    for (i = 0; i < plan->N_target; i++) {
        indices_n[i] = 0;
        values[i] = 0;
    }


    for (i = 0; i < plan->N_target; i++) {

    	bulid_indices_single_II(plan, plan->tree, i, indices_n, values, 0);

        for (j = 0; j < plan->N_target; j++) {
            if (indices_n[j] > 0) {
                total_length_n++;
                indices_n[j] = 0;
            }
        }

    }

    plan->total_length_n=total_length_n;

    plan->id_n = malloc(total_length_n * sizeof ( int));
    plan->b_m = malloc(total_length_n * sizeof ( double));

    for (i = 0; i < total_length_n; i++) {
        plan->id_n[i] = 0;
        plan->b_m[i] = 0;
    }


    total_length_n = 0;
    for (i = 0; i < plan->N_target; i++) {

    	bulid_indices_single_II(plan, plan->tree, i, indices_n, values, 1);

        tmp_length_n = 0;

        for (j = 0; j < plan->N_target; j++) {
            if (indices_n[j] > 0) {
                plan->id_n[total_length_n] = j;
                plan->b_m[total_length_n] = values[j];

                total_length_n++;
                tmp_length_n++;
                indices_n[j] = 0;
                values[j] = 0;
            }
        }

        plan->id_nn[i] = tmp_length_n;
    }


    free(indices_n);
    free(values);

}


inline double direct_compute_potential_leaf(fastsum_plan *plan, struct octree_node *tree, int index){

	int i,k;
	double res=0;
	double dx,dy,dz,R;

	for (i = tree->begin; i < tree->end; i++) {

		dx = plan->x_t[3 * index]-plan->x_s[3 * i];
	    dy = plan->x_t[3 * index + 1]-plan->x_s[3 * i + 1];
	    dz = plan->x_t[3 * index + 2]-plan->x_s[3 * i + 2];

	    R = dx * dx + dy * dy + dz * dz;

	    if (R>plan->r_eps_squre){

	    	k = plan->x_s_ids[i];
	    	dx *= plan->t_normal[3 * k];
	    	dy *= plan->t_normal[3 * k + 1];
	        dz *= plan->t_normal[3 * k + 2];

	        res += plan->charge_density[k]*(dx + dy + dz) / (R*sqrt(R));

	    }
	}

	return res;

}


double compute_potential_single_target_II(fastsum_plan *plan, struct octree_node *tree, int index, double ***a) {

    double R,r;
    int i, j, k;
    double res = 0;
    double dx, dy, dz;


    R = pow2(plan->x_t[3 * index] - tree->x)
            + pow2(plan->x_t[3 * index + 1] - tree->y)
            + pow2(plan->x_t[3 * index + 2] - tree->z);

    r = sqrt(R);

    if (plan->mac * r > tree->radius) {

        if (plan->r_eps>r*(1-plan->mac)){

        	res=direct_compute_potential_leaf(plan,tree,index);

        	return res;

        }

        if (!tree->have_moment) {
                tree->moment = alloc_3d_double(plan->p + 1, plan->p + 1, plan->p + 1);
                tree->have_moment = 1;
                tree->need_upadte_moment = 1;
        }


        if (tree->need_upadte_moment) {
            compute_moment(plan, tree, tree->moment, tree->x, tree->y, tree->z);
            tree->need_upadte_moment = 0;
        }


        dx = plan->x_t[3 * index] - tree->x;
        dy = plan->x_t[3 * index + 1] - tree->y;
        dz = plan->x_t[3 * index + 2] - tree->z;
        compute_coefficient(a, dx, dy, dz, plan->p);

        for (i = 0; i < plan->p + 1; i++) {
            for (j = 0; j < plan->p - i + 1; j++) {
                for (k = 0; k < plan->p - i - j + 1; k++) {
                    res += a[i][j][k] * tree->moment[i][j][k];
                }
            }
        }


        return res;

    } else {

        if (tree->num_children > 0) {

        	res = 0;
            for (i = 0; i < tree->num_children; i++) {
                res += compute_potential_single_target_II(plan, tree->children[i], index, a);
            }
            return res;

        } else {

        	res=direct_compute_potential_leaf(plan,tree,index);

        	return res;
        }
    }

}


void compute_analytical_potential(fastsum_plan *plan, double *phi, double *u1) {

	int i, j, k;
    int total_j = 0;

    for (i = 0; i < plan->N_target; i++) {

        for (j = 0; j < plan->id_nn[i]; j++) {
            k = plan->id_n[total_j];
            phi[i] += plan->b_m[total_j] * u1[k];
            total_j++;
        }

        phi[i]+=plan->vert_bsa[i] * u1[i];

    }

}



void direct_sum_I(fastsum_plan *plan, double *phi, double *u1) {
    int i, j, k;
    double dx,dy,dz;
    double res,R;

    for (j = 0; j < plan->N_target; j++) {
    	 res=0;
    	 phi[j]=0;

         for(i=0;i<plan->N_source;i++){

        	 dx = plan->x_t[3 * j]-plan->x_s[3 * i];
        	 dy = plan->x_t[3 * j + 1]-plan->x_s[3 * i + 1];
        	 dz = plan->x_t[3 * j + 2]-plan->x_s[3 * i + 2];

        	 R = dx * dx + dy * dy + dz * dz;

        	 if (R>plan->r_eps_squre){

        	 	 k = plan->x_s_ids[i];
        	 	 dx *= plan->t_normal[3 * k];
        	 	 dy *= plan->t_normal[3 * k + 1];
        	 	 dz *= plan->t_normal[3 * k + 2];

        	 	 res += plan->charge_density[k]*(dx + dy + dz) / (R*sqrt(R));

        	 }

         }

         phi[j]=res;

    }

    compute_analytical_potential(plan,phi,u1);

}


void fast_sum_II(fastsum_plan *plan, double *phi, double *u1) {

	int i,j;

    double ***a = alloc_3d_double(plan->p + 1, plan->p + 1, plan->p + 1);

    for (j = 0; j < plan->N_target; j++) {

         phi[j] = compute_potential_single_target_II(plan, plan->tree, j, a);

     }

    compute_analytical_potential(plan,phi,u1);

    free_3d_double(a, plan->p + 1);

}

