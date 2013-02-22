#include "treecode_bem_helper.h"



void init_fastsum_II(fastsum_plan *plan, int N_target, int triangle_num, int p, double mac, int num_limit, double correct_factor) {

	init_fastsum_I(plan,N_target,triangle_num,p,mac,num_limit);

    plan->id_tn = (int *) malloc(N_target * (sizeof (int)));

    plan->r_eps_factor=correct_factor;


}


void bulid_indices_single_II(fastsum_plan *plan, struct octree_node *tree,
        int index, int *it, int *in, double *value, int compute_bm) {

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
            bulid_indices_single_II(plan, tree->children[i], index, it, in, value, compute_bm);
        }
        return;

    } else {


        for (i = tree->begin; i < tree->end; i++) {

            dx = plan->x_s[3 * i] - plan->x_t[3 * index];
            dy = plan->x_s[3 * i + 1] - plan->x_t[3 * index + 1];
            dz = plan->x_s[3 * i + 2] - plan->x_t[3 * index + 2];

            r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r < plan->r_eps) {
                it[i] = 1;

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

    int *indices_t = malloc(plan->N_source * sizeof ( int));
    int *indices_n = malloc(plan->N_target * sizeof ( int));
    double *values = malloc(plan->N_target * sizeof ( double));

    int tmp_length_t = 0;
    int tmp_length_n = 0;
    int total_length_t = 0;
    int total_length_n = 0;

    for (i = 0; i < plan->N_target; i++) {
        indices_n[i] = 0;
        values[i] = 0;
    }

    for (i = 0; i < plan->N_source; i++) {
        indices_t[i] = 0;
    }

    for (i = 0; i < plan->N_target; i++) {

        bulid_indices_single_II(plan, plan->tree, i, indices_t, indices_n, values, 0);

        for (j = 0; j < plan->N_source; j++) {
            if (indices_t[j] > 0) {
                total_length_t++;
                indices_t[j] = 0;
            }
        }

        for (j = 0; j < plan->N_target; j++) {
            if (indices_n[j] > 0) {
                total_length_n++;
                indices_n[j] = 0;
            }
        }

    }

    plan->id_t = malloc(total_length_t * sizeof ( int));
    plan->id_n = malloc(total_length_n * sizeof ( int));
    plan->b_m = malloc(total_length_n * sizeof ( double));


    total_length_t = 0;
    total_length_n = 0;
    for (i = 0; i < plan->N_target; i++) {
        plan->id_tn[i] = 0;
        plan->id_nn[i] = 0;

        bulid_indices_single_II(plan, plan->tree, i, indices_t, indices_n, values, 1);

        tmp_length_t = 0;
        tmp_length_n = 0;

        for (j = 0; j < plan->N_source; j++) {
            if (indices_t[j] > 0) {
                plan->id_t[total_length_t] = j;
                total_length_t++;
                tmp_length_t++;
                indices_t[j] = 0;
            }
        }

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

        plan->id_tn[i] = tmp_length_t;
        plan->id_nn[i] = tmp_length_n;
    }

    plan->total_length_n=total_length_n;

    /*
    for (i = 0; i < plan->N_target; i++) {
        printf("%d = %d   %d\n", i, plan->id_tn[i], plan->id_nn[i]);
    }

    for (i = 0; i < total_length_t; i++) {
        printf("id_t[%d] =    %d\n", i, plan->id_t[i]);
    }
    */

    free(indices_t);
    free(indices_n);
    free(values);

}



double compute_potential_single_target_II(fastsum_plan *plan, struct octree_node *tree, int index, double ***a) {

    double R;
    int i, j, k;
    double res = 0;
    double dx, dy, dz;


    R = pow2(plan->x_t[3 * index] - tree->x)
            + pow2(plan->x_t[3 * index + 1] - tree->y)
            + pow2(plan->x_t[3 * index + 2] - tree->z);


    if (plan->mac_square * R > tree->radius_square) {
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

        	res=0;
        	for (i = tree->begin; i < tree->end; i++) {
        		k = plan->x_s_ids[i];
        	    dx = plan->x_t[3 * index]-plan->x_s[3 * i];
        	    dy = plan->x_t[3 * index + 1]-plan->x_s[3 * i + 1];
        	    dz = plan->x_t[3 * index + 2]-plan->x_s[3 * i + 2];

        	    R = dx * dx + dy * dy + dz * dz;

        	    dx *= plan->t_normal[3 * k];
        	    dy *= plan->t_normal[3 * k + 1];
        	    dz *= plan->t_normal[3 * k + 2];

        	    res += plan->charge_density[k]*(dx + dy + dz) / (R*sqrt(R));

        	}

        	return res;
        }
    }

}


void compute_correction(fastsum_plan *plan, double *phi, double *u1) {
    int i, j, k, t;
    double dx,dy,dz;
    double R;
    int total_j = 0;
    int total_t = 0;

    for (i = 0; i < plan->N_target; i++) {

        for (j = 0; j < plan->id_tn[i]; j++) {

        	k=plan->id_t[total_t];

            dx = plan->x_t[3 * i]-plan->x_s[3 * k];
            dy = plan->x_t[3 * i + 1]-plan->x_s[3 * k + 1];
            dz = plan->x_t[3 * i + 2]-plan->x_s[3 * k + 2];

            R = dx * dx + dy * dy + dz * dz;

            t = plan->x_s_ids[k];

            dx *= plan->t_normal[3 * t];
      	    dy *= plan->t_normal[3 * t + 1];
      	    dz *= plan->t_normal[3 * t + 2];

      	    phi[i] -= plan->charge_density[t]*(dx + dy + dz) / (R*sqrt(R));

      	    total_t++;
        }


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

        	 k = plan->x_s_ids[i];
             dx = plan->x_t[3 * j]-plan->x_s[3 * i];
             dy = plan->x_t[3 * j + 1]-plan->x_s[3 * i + 1];
             dz = plan->x_t[3 * j + 2]-plan->x_s[3 * i + 2];

             R = dx * dx + dy * dy + dz * dz;

             dx *= plan->t_normal[3 * k];
             dy *= plan->t_normal[3 * k + 1];
             dz *= plan->t_normal[3 * k + 2];

             res += plan->charge_density[k]*(dx + dy + dz) / (R*sqrt(R));

         }

         phi[j]=res;

    }

    compute_correction(plan,phi,u1);

}


void fast_sum_II(fastsum_plan *plan, double *phi, double *u1) {

	int j;

    double ***a = alloc_3d_double(plan->p + 1, plan->p + 1, plan->p + 1);

    for (j = 0; j < plan->N_target; j++) {

         phi[j] = compute_potential_single_target_II(plan, plan->tree, j, a);

     }

     free_3d_double(a, plan->p + 1);

     compute_correction(plan,phi,u1);

}

