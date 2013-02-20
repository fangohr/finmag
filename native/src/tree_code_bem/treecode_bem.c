#include "treecode_bem.h"



inline double pow2(double x) {
    return x*x;
}

inline double distance(double *x, double *y) {
    double r = 0;
    r = (x[0] - y[0])*(x[0] - y[0])
            +(x[1] - y[1])*(x[1] - y[1])
            +(x[2] - y[2])*(x[2] - y[2]);
    return sqrt(r);
}

inline double tri_max(double x, double y, double z) {

    double tmp = x > y ? x : y;

    x = tmp > z ? tmp : z;

    return x;
}

inline double tri_min(double x, double y, double z) {

    double tmp = x < y ? x : y;

    x = tmp < z ? tmp : z;

    return x;
}

double array_max(double *x, int N, int t) {
    int i;
    double max = x[t];
    for (i = 0; i < N; i++) {
        if (max < x[3 * i + t]) {
            max = x[3 * i + t];
        }
    }

    return max;
}

double array_min(double *x, int N, int t) {
    int i;
    double min = x[t];
    for (i = 0; i < N; i++) {
        if (min > x[3 * i + t]) {
            min = x[3 * i + t];
        }
    }

    return min;
}

//compute det(J1),det(J2),det(J3) (equations are given in the report)

inline double det2(double *dx, double *dy, double *dz) {

    double J1, J2, J3, res;

    J1 = (dy[0] - dx[0])*(dz[1] - dx[1])-(dy[1] - dx[1])*(dz[0] - dx[0]);

    J2 = (dy[0] - dx[0])*(dz[2] - dx[2])-(dy[2] - dx[2])*(dz[0] - dx[0]);

    J3 = (dy[1] - dx[1])*(dz[2] - dx[2])-(dy[2] - dx[2])*(dz[1] - dx[1]);

    res = sqrt(J1 * J1 + J2 * J2 + J3 * J3);

    return res;

}

inline double det(double *x, double *y, double *z) {
    double d;
    d = x[0] * y[1] * z[2] + x[1] * y[2] * z[0] \
      + x[2] * y[0] * z[1] - x[0] * y[2] * z[1] \
      - x[1] * y[0] * z[2] - x[2] * y[1] * z[0];
    return d;
}

inline double dot(double *x, double *y) {
    double res;
    res = x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
    return res;
}

inline double norm(double *p) {

    double tmp = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];

    return sqrt(tmp);

}

void vector_unit(double *p, double *up) {

    double tmp = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];

    tmp = sqrt(tmp);
    tmp = 1.0 / tmp;

    up[0] = p[0] * tmp;
    up[1] = p[1] * tmp;
    up[2] = p[2] * tmp;
}

inline void cross(double *x, double *y, double *z) {
    z[0] = x[1] * y[2] - x[2] * y[1];
    z[1] = -x[0] * y[2] + x[2] * y[0];
    z[2] = x[0] * y[1] - x[1] * y[0];
}

double solid_angle_single(double *p, double *x1, double *x2, double *x3) {

    int i;
    double x[3], y[3], z[3];
    double d, a, b, c;
    double omega, div;

    for (i = 0; i < 3; i++) {
        x[i] = x1[i] - p[i];
        y[i] = x2[i] - p[i];
        z[i] = x3[i] - p[i];
    }

    d = det(x, y, z);
    a = norm(x);
    b = norm(y);
    c = norm(z);

    div = a * b * c + dot(x, y) * c + dot(x, z) * b + dot(y, z) * a;
    omega = atan2(fabs(d), div);

    if (omega < 0) {
        omega += M_PI;
    }

    return (2 * omega);
}

void boundary_element(double *xp, double *x1, double *x2, double *x3, double *res) {

    int i;
    double zetav[3], zeta;
    double xi1[3], xi2[3], xi3[3];
    double sv1[3], sv2[3], sv3[3];
    double rhov1[3], rhov2[3], rhov3[3];
    double etav1[3], etav2[3], etav3[3];
    double gamma1[3], gamma2[3], gamma3[3];
    double s1, s2, s3, s, area, p[3];
    double eta1, eta2, eta3;
    double rho1, rho2, rho3;

    double omega, tmp[3];

    omega = solid_angle_single(xp, x1, x2, x3);

    if (omega == 0) {
        res[0] = 0;
        res[1] = 0;
        res[2] = 0;
        return;
    }

    for (i = 0; i < 3; i++) {
        rhov1[i] = x1[i] - xp[i];
        rhov2[i] = x2[i] - xp[i];
        rhov3[i] = x3[i] - xp[i];
        sv1[i] = x2[i] - x1[i];
        sv2[i] = x3[i] - x2[i];
        sv3[i] = x1[i] - x3[i];
    }

    cross(sv1, sv2, zetav);

    vector_unit(sv1, xi1);
    vector_unit(sv2, xi2);
    vector_unit(sv3, xi3);
    vector_unit(zetav, zetav);

    zeta = dot(zetav, rhov1);
    cross(zetav, xi1, etav1);
    cross(zetav, xi2, etav2);
    cross(zetav, xi3, etav3);

    eta1 = dot(etav1, rhov1);
    eta2 = dot(etav2, rhov2);
    eta3 = dot(etav3, rhov3);


    gamma1[0] = dot(xi2, xi1);
    gamma1[1] = dot(xi2, xi2);
    gamma1[2] = dot(xi2, xi3);

    gamma2[0] = dot(xi3, xi1);
    gamma2[1] = dot(xi3, xi2);
    gamma2[2] = dot(xi3, xi3);

    gamma3[0] = dot(xi1, xi1);
    gamma3[1] = dot(xi1, xi2);
    gamma3[2] = dot(xi1, xi3);


    s1 = norm(sv1);
    s2 = norm(sv2);
    s3 = norm(sv3);
    s = (s1 + s2 + s3) / 2.0;
    area = sqrt(s * (s - s1)*(s - s2)*(s - s3));


    rho1 = norm(rhov1);
    rho2 = norm(rhov2);
    rho3 = norm(rhov3);

    p[0] = log((rho1 + rho2 + s1) / (rho1 + rho2 - s1));
    p[1] = log((rho2 + rho3 + s2) / (rho2 + rho3 - s2));
    p[2] = log((rho3 + rho1 + s3) / (rho3 + rho1 - s3));

    for (i = 0; i < 3; i++) {
        tmp[i] = 0;
    }

    for (i = 0; i < 3; i++) {
        tmp[0] += gamma1[i] * p[i];
        tmp[1] += gamma2[i] * p[i];
        tmp[2] += gamma3[i] * p[i];
    }

    if (zeta < 0) {
        omega = -omega;
    }

    res[0] = (eta2 * omega - zeta * tmp[0]) * s2 / (8.0 * M_PI * area);
    res[1] = (eta3 * omega - zeta * tmp[1]) * s3 / (8.0 * M_PI * area);
    res[2] = (eta1 * omega - zeta * tmp[2]) * s1 / (8.0 * M_PI * area);

    return;
}


void compute_source_nodes_weights(fastsum_plan *plan) {
    int face, f, k;

    double x1, y1, z1;
    double x2, y2, z2;
    double x3, y3, z3;
    double v, *p1, *p2, *p3;
    double total_area=0;

    for (face = 0; face < plan->triangle_num; face++) {

        f = 3 * face;

        k = plan->triangle_nodes[f];
        p1 = &plan->x_t[3 * k];
        x1 = p1[0];
        y1 = p1[1];
        z1 = p1[2];

        k = plan->triangle_nodes[f + 1];
        p2 = &plan->x_t[3 * k];
        x2 = p2[0] - x1;
        y2 = p2[1] - y1;
        z2 = p2[2] - z1;

        k = plan->triangle_nodes[f + 2];
        p3 = &plan->x_t[3 * k];
        x3 = p3[0] - x1;
        y3 = p3[1] - y1;
        z3 = p3[2] - z1;

        v = det2(p1, p2, p3);

        plan->x_s_tri[f] = x2 / 3.0 + x3 / 3.0 + x1;
        plan->x_s_tri[f + 1] = y2 / 3.0 + y3 / 3.0 + y1;
        plan->x_s_tri[f + 2] = z2 / 3.0 + z3 / 3.0 + z1;

        plan->x_s[f] = plan->x_s_tri[f];
        plan->x_s[f + 1] = plan->x_s_tri[f+1];
        plan->x_s[f + 2] = plan->x_s_tri[f+2];

        plan->weights[face] = v / 2.0 / (4 * M_PI);

        total_area+=v/2.0;
    }

    total_area=total_area/plan->triangle_num*2.0/sqrt(3);
    plan->r_eps=sqrt(total_area)*plan->r_eps_factor;
    printf("plan->r_eps=%g   factor=%g\n",plan->r_eps,plan->r_eps_factor);

}


double **alloc_2d_double(int ndim1, int ndim2) {

    int i;

    double **array2 = malloc(ndim1 * sizeof ( double *));

    if (array2 != NULL) {

        array2[0] = malloc(ndim1 * ndim2 * sizeof ( double));

        if (array2[0] != NULL) {

            for (i = 1; i < ndim1; i++)
                array2[i] = array2[0] + i * ndim2;

        } else {
            free(array2);
            array2 = NULL;
        }

    }

    return array2;

}

void free_2d_double(double **p) {

    free(p[0]);
    free(p);

}

double ***alloc_3d_double(int ndim1, int ndim2, int ndim3) {

    double *space = malloc(ndim1 * ndim2 * ndim3 * sizeof ( double));

    double ***array3 = malloc(ndim1 * sizeof ( double**));

    int i, j;

    memset(space, 0, ndim1 * ndim2 * ndim3 * sizeof ( double));

    if (space == NULL || array3 == NULL)
        return NULL;

    for (j = 0; j < ndim1; j++) {
        array3[ j ] = malloc(ndim2 * sizeof ( double*));

        if (array3[ j ] == NULL)
            return NULL;
        for (i = 0; i < ndim2; i++)
            array3[ j ][ i ] = space + j * (ndim3 * ndim2) + i * ndim3;
    }

    return array3;

}

void free_3d_double(double ***p, int ndim1) {

    int i;

    free(p[0][0]);

    for (i = 0; i < ndim1; i++)
        free(p[i]);

    free(p);

}

//helper function 

void quicksort(int t, double *x, int *index, int N) {
    int lpos = 0;
    int rpos = N - 1;

    double pivot = x[(N / 2)*3 + t];

    int k;
    double temp1;
    int tmp_index;

    while (lpos <= rpos) {
        while (x[3 * lpos + t] < pivot)
            lpos++;
        while (x[3 * rpos + t] > pivot)
            rpos--;
        if (lpos <= rpos) {
            for (k = 0; k < 3; k++) {
                temp1 = x[3 * lpos + k];
                x[3 * lpos + k] = x[3 * rpos + k];
                x[3 * rpos + k] = temp1;
            }
            tmp_index = index[lpos];
            index[lpos] = index[rpos];
            index[rpos] = tmp_index;

            lpos++;
            rpos--;
        }
    }
    if (0 < rpos)
        quicksort(t, x, index, rpos + 1);
    if (lpos < N - 1)
        quicksort(t, x + 3 * lpos, index + lpos, N - lpos);
}

void bulid_indices_single(fastsum_plan *plan, struct octree_node *tree,
        int index, int *it, int *in, double *value, int compute_bm) {

    double r, rc;
    int i, j, k;
    double dx, dy, dz;
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
            bulid_indices_single(plan, tree->children[i], index, it, in, value, compute_bm);
        }
        return;

    } else {


        for (i = tree->begin; i < tree->end; i++) {

            dx = plan->x_s_tri[3 * i] - plan->x_t[3 * index];
            dy = plan->x_s_tri[3 * i + 1] - plan->x_t[3 * index + 1];
            dz = plan->x_s_tri[3 * i + 2] - plan->x_t[3 * index + 2];

            r = sqrt(dx * dx + dy * dy + dz * dz);

            if (r < plan->r_eps) {
                it[i] = 1;

                j = plan->index_tri[i];
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

void bulid_indices(fastsum_plan *plan) {

    int i, j, k;

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

        if (plan->g2b[i] < 0) {
            continue;
        }


        bulid_indices_single(plan, plan->tree, i, indices_t, indices_n, values, 0);

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

    //printf("total_length_t=%d   total_length_n=%d\n", total_length_t, total_length_n);


    plan->id_t = malloc(total_length_t * sizeof ( int));
    plan->id_n = malloc(total_length_n * sizeof ( int));
    plan->b_m = malloc(total_length_n * sizeof ( double));


    total_length_t = 0;
    total_length_n = 0;
    for (i = 0; i < plan->N_target; i++) {
        plan->id_tn[i] = 0;
        plan->id_nn[i] = 0;

        if (plan->g2b[i] < 0) {
            continue;
        }

        bulid_indices_single(plan, plan->tree, i, indices_t, indices_n, values, 1);

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


int get_total_length(fastsum_plan *plan){
	printf("total_length=%d \n",plan->total_length_n);
    return plan->total_length_n;
}


int divide_box(fastsum_plan *plan, struct octree_node *tree, double *bnd, double **bnds, int bend[8][2]) {

    int i, j, k;

    int children_box_num = 1;

    double max_size = tri_max(tree->rx, tree->ry, tree->rz);

    double critial = max_size / sqrt(2.0);


    for (i = 0; i < 8; i++) {
        bend[i][0] = tree->begin;
        bend[i][1] = tree->end;
    }

    if (tree->rx > critial) {

        for (j = 0; j < 6; j++) {
            bnds[0][j] = bnd[j];
            bnds[1][j] = bnd[j];
        }

        bnds[0][0] = bnd[0];
        bnds[0][1] = (bnd[0] + bnd[1]) / 2.0;
        bnds[1][0] = (bnd[0] + bnd[1]) / 2.0;
        bnds[1][1] = bnd[1];

        quicksort(0, plan->x_s_tri + 3 * tree->begin, plan->index_tri + tree->begin, tree->num_particle);

        for (k = tree->begin; k < tree->end; k++) {
            if (plan->x_s_tri[3 * k] >= bnds[0][1])
                break;
        }

        bend[0][0] = tree->begin;
        bend[0][1] = k;

        bend[1][0] = k;
        bend[1][1] = tree->end;

        children_box_num *= 2;
        //printf("x-direction\n");
    }



    if (tree->ry > critial) {

        for (i = 0; i < children_box_num; i++) {

            for (j = 0; j < 6; j++) {
                bnds[children_box_num + i][j] = bnds[i][j];
            }

            bnds[i][2] = bnd[2];
            bnds[i][3] = (bnd[2] + bnd[3]) / 2.0;
            bnds[children_box_num + i][2] = (bnd[2] + bnd[3]) / 2.0;
            bnds[children_box_num + i][3] = bnd[3];

            //printf("bend[%d][0]=%d  %d\n", i, bend[i][0], bend[i][1]);

            quicksort(1, plan->x_s_tri + 3 * bend[i][0], plan->index_tri + bend[i][0], bend[i][1] - bend[i][0]);

            for (k = bend[i][0]; k < bend[i][1]; k++) {
                if (plan->x_s_tri[3 * k + 1] >= bnds[i][3])
                    break;
            }


            bend[children_box_num + i][0] = k;
            bend[children_box_num + i][1] = bend[i][1];
            bend[i][1] = k;

            //printf("y-direction %d  %d  %d \n", bend[i][1], bend[children_box_num + i][0], bend[children_box_num + i][1]);

        }

        children_box_num *= 2;

    }


    if (tree->rz > critial) {

        for (i = 0; i < children_box_num; i++) {

            for (j = 0; j < 6; j++) {
                bnds[children_box_num + i][j] = bnds[i][j];
            }

            bnds[i][4] = bnd[4];
            bnds[i][5] = (bnd[4] + bnd[5]) / 2.0;
            bnds[children_box_num + i][4] = (bnd[4] + bnd[5]) / 2.0;
            bnds[children_box_num + i][5] = bnd[5];

            quicksort(2, plan->x_s_tri + 3 * bend[i][0], plan->index_tri + bend[i][0], bend[i][1] - bend[i][0]);

            for (k = bend[i][0]; k < bend[i][1]; k++) {
                if (plan->x_s_tri[3 * k + 2] >= bnds[i][5])
                    break;
            }


            bend[children_box_num + i][0] = k;
            bend[children_box_num + i][1] = bend[i][1];
            bend[i][1] = k;

        }

        children_box_num *= 2;
        //printf("z-direction\n");
    }


    return children_box_num;
}

void create_tree(fastsum_plan *plan, struct octree_node *tree, int begin, int end, double *bnd) {

    double **bnds;
    int bend[8][2]; //begin and end
    int children_num = 0;
    int possible_children_num;
    int i;


    tree->have_moment = 0;
    tree->moment = NULL;

    tree->begin = begin;
    tree->end = end;
    tree->num_particle = end - begin;

    plan->tree->have_moment = 0;

    tree->rx = (bnd[1] - bnd[0]) / 2.0;
    tree->ry = (bnd[3] - bnd[2]) / 2.0;
    tree->rz = (bnd[5] - bnd[4]) / 2.0;

    tree->x = (bnd[1] + bnd[0]) / 2.0;
    tree->y = (bnd[3] + bnd[2]) / 2.0;
    tree->z = (bnd[5] + bnd[4]) / 2.0;


    tree->radius_square = pow2(tree->rx)
            + pow2(tree->ry) + pow2(tree->rz);

    tree->radius=sqrt(tree->radius_square);

    tree->num_children = 0;


    if (tree->num_particle > plan->num_limit) {

        bnds = alloc_2d_double(8, 6);

        possible_children_num = divide_box(plan, tree, bnd, bnds, bend);

        children_num = 0;
        for (i = 0; i < possible_children_num; i++) {

            if (bend[i][1] - bend[i][0] > 0) {

                tree->children[children_num] = (struct octree_node *) malloc(sizeof (struct octree_node));

                create_tree(plan, tree->children[children_num], bend[i][0], bend[i][1], bnds[i]);

                children_num++;
            }

        }

        tree->num_children = children_num;
        free_2d_double(bnds);
    }
}

void printree(struct octree_node *tree) {
    int i;
    if (tree->num_children > 0) {
        for (i = 0; i < tree->num_children; i++) {
            printree(tree->children[i]);
        }
    } else {
        printf("xyz=%f  %f  %f children=%d  begin %d, %d  radius=%g\n",
                tree->x, tree->y, tree->z, tree->num_children,
                tree->begin, tree->end, sqrt(tree->radius_square));
    }

}

void free_tree(fastsum_plan *plan, struct octree_node *tree) {
    int i;
    if (tree == NULL) {
        return;
    }
    if (tree->num_children > 0) {
        for (i = 0; i < tree->num_children; i++) {
            free_tree(plan, tree->children[i]);
        }
    } else {
        if (tree->moment != NULL) {
            free_3d_double(tree->moment, plan->p + 1);
        }
        free(tree);
    }

}

void build_tree(fastsum_plan *plan) {

    double bnd[6];

    plan->tree = (struct octree_node *) malloc(sizeof (struct octree_node));

    bnd[0] = array_min(plan->x_s_tri, plan->triangle_num, 0);
    bnd[1] = array_max(plan->x_s_tri, plan->triangle_num, 0);

    bnd[2] = array_min(plan->x_s_tri, plan->triangle_num, 1);
    bnd[3] = array_max(plan->x_s_tri, plan->triangle_num, 1);

    bnd[4] = array_min(plan->x_s_tri, plan->triangle_num, 2);
    bnd[5] = array_max(plan->x_s_tri, plan->triangle_num, 2);


    //printf("start to build tree  %d\n", plan->N_source);
    create_tree(plan, plan->tree, 0, plan->triangle_num, bnd);
    //printree(plan->tree);
    //printf("build tree okay\n");
    /*
    int i,k;
    double dx,dy,dz;

    for(i=0;i<plan->N_source;i++){
    	k = plan->index_tri[i];
    	dx = plan->x_s[3*i]-plan->x_s_tri[3 * k];
    	dy = plan->x_s[3 * i + 1]-plan->x_s_tri[3 * k + 1];
    	dz = plan->x_s[3 * i + 2]-plan->x_s_tri[3 * k + 2];

    	printf("r1=%g   ",sqrt(dx*dx+dy*dy+dz*dz));

    	dx = plan->x_s[3*k]-plan->x_s_tri[3 * i];
    	dy = plan->x_s[3 * k + 1]-plan->x_s_tri[3 * i + 1];
    	dz = plan->x_s[3 * k + 2]-plan->x_s_tri[3 * i + 2];
    	printf("r2=%g\n",sqrt(dx*dx+dy*dy+dz*dz));

    }
    */

}

void compute_Taylor(double ***a, double dx, double dy, double dz, int p) {
    int i, j, k;
    double R, r, r3, r5;

    double *cf = (double *) malloc((p + 1) * sizeof (double));
    double *cf2 = (double *) malloc((p + 1) * sizeof (double));

    double ddx = 2 * dx;
    double ddy = 2 * dy;
    double ddz = 2 * dz;

    R = 1.0 / (dx * dx + dy * dy + dz * dz);

    r = sqrt(R);
    r3 = r*R;
    r5 = r3*R;

    a[0][0][0] = r;

    a[1][0][0] = dx*r3;
    a[0][1][0] = dy*r3;
    a[0][0][1] = dz*r3;

    a[1][1][0] = 3 * dx * dy*r5;
    a[1][0][1] = 3 * dx * dz*r5;
    a[0][1][1] = 3 * dy * dz*r5;


    for (i = 1; i < p + 1; i++) {
        cf[i] = 1 - 1.0 / i;
        cf2[i] = 1 - 0.5 / i;
    }


    for (i = 2; i < p + 1; i++) {
        a[i][0][0] = (2 * dx * cf2[i] * a[i - 1][0][0]
                - cf[i] * a[i - 2][0][0]) * R;

        a[0][i][0] = (2 * dy * cf2[i] * a[0][i - 1][ 0]
                - cf[i] * a[0][ i - 2][0]) * R;

        a[0][0][i] = (2 * dz * cf2[i] * a[0][0][i - 1]
                - cf[i] * a[0][0][ i - 2]) * R;
    }


    for (i = 2; i < p + 1; i++) {
        a[1][0][i] = (dx * a[0][0][i] + ddz * a[1][0][ i - 1] - a[1][0][i - 2]) * R;
        a[0][1][i] = (dy * a[0][0][i] + ddz * a[0][1][ i - 1] - a[0][1][i - 2]) * R;
        a[0][i][1] = (dz * a[0][i][0] + ddy * a[0][i - 1][1] - a[0][i - 2][1]) * R;
        a[1][i][0] = (dx * a[0][i][0] + ddy * a[1][i - 1][0] - a[1][i - 2][0]) * R;
        a[i][1][0] = (dy * a[i][0][0] + ddx * a[i - 1][1][0] - a[i - 2][1][0]) * R;
        a[i][0][1] = (dz * a[i][0][0] + ddx * a[i - 1][0][1] - a[i - 2][0][1]) * R;
    }


    for (i = 2; i < p - 1; i++) {
        for (j = 2; j < p - i + 1; j++) {

            a[i][j][0] = (ddx * cf2[i] * a[i - 1][j][0] + ddy * a[i][j - 1][0]
                    - cf[i] * a[i - 2][j][0] - a[i][j - 2][0]) * R;

            a[i][0][j] = (ddx * cf2[i] * a[i - 1][0][j] + ddz * a[i][0][j - 1]
                    - cf[i] * a[i - 2][0][j] - a[i][0][j - 2]) * R;

            a[0][i][j] = (ddy * cf2[i] * a[0][i - 1][j] + ddz * a[0][i][j - 1]
                    - cf[i] * a[0][i - 2][j] - a[0][i][j - 2]) * R;

        }
    }

    for (i = 2; i < p - 2; i++) {
        for (j = 2; j < p - i; j++) {
            a[1][i][j] = (dx * a[0][i][j] + ddy * a[1][ i - 1][j] + ddz * a[1][i][j - 1]
                    - a[1][ i - 2][j] - a[1][i][j - 2]) * R;
            a[i][1][j] = (dy * a[i][0][j] + ddx * a[i - 1][1][j] + ddz * a[i][1][ j - 1]
                    - a[i - 2][1][j] - a[i][1][j - 2]) * R;
            a[i][j][1] = (dz * a[i][j][0] + ddy * a[i - 1][j][1] + ddy * a[i][j - 1][1]
                    - a[i - 2][j][1] - a[i][j - 2][1]) * R;
        }
    }


    for (i = 2; i < p - 3; i++) {
        for (j = 2; j < p - i - 1; j++) {
            for (k = 2; k < p - i - j + 1; k++) {

                a[i][j][k] = (ddx * cf2[i] * a[i - 1][j][ k] + ddy * a[i][j - 1][0]
                        + 2 * dz * a[i][j][k - 1] - cf[i] * a[i - 2][j][k]
                        - a[i][j - 2][k] - a[i][ j][ k - 2]) * R;

            }
        }
    }

    free(cf);
    free(cf2);

    return;
}

void compute_moment(fastsum_plan *plan, struct octree_node *tree, double ***moment, double x, double y, double z) {
    double dx, dy, dz;
    int i, j, k, ti, tj, tk, tn;
    double tmp_x, tmp_y, tmp_z;
    double nx, ny, nz;
    double tmp_charge;
    //double tmp_moment = 0;



    for (i = 0; i < plan->p + 1; i++) {
        for (j = 0; j < plan->p - i + 1; j++) {
            for (k = 0; k < plan->p - i - j + 1; k++) {
                moment[i][j][k] = 0;
            }
        }
    }

    for (ti = tree->begin; ti < tree->end; ti++) {

        tj = plan->index_tri[ti];

        nx = plan->t_normal[3 * tj];
        ny = plan->t_normal[3 * tj + 1];
        nz = plan->t_normal[3 * tj + 2];

        dx = plan->x_s_tri[3 * ti] - x;
        dy = plan->x_s_tri[3 * ti + 1] - y;
        dz = plan->x_s_tri[3 * ti + 2] - z;

        if (dx==0){
        	dx=1;
        	nx=0;
        }

        if (dy==0){
        	dy=1;
        	ny=0;
        }

        if (dz==0){
        	dz=1;
        	nz=0;
        }

        tmp_charge = plan->charge_density[tj];

        tmp_x = 1.0;
        for (i = 0; i < plan->p + 1; i++) {
             tmp_y = 1.0;
             for (j = 0; j < plan->p - i + 1; j++) {
                    tmp_z = 1.0;
                    for (k = 0; k < plan->p - i - j + 1; k++) {
                        moment[i][j][k] += tmp_charge * (
                             i * tmp_x * tmp_y * tmp_z / dx * nx +
                             j * tmp_x * tmp_y * tmp_z / dy * ny +
                             k * tmp_x * tmp_y * tmp_z / dz * nz);
                        tmp_z *= dz;
                    }
                    tmp_y *= dy;
                }
                tmp_x *= dx;
            }
    }


}

double compute_potential_single_target(fastsum_plan *plan, struct octree_node *tree, int index, double ***a) {

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
        compute_Taylor(a, dx, dy, dz, plan->p);

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
                res += compute_potential_single_target(plan, tree->children[i], index, a);
            }
            return res;

        } else {

        	res=0;
        	for (i = tree->begin; i < tree->end; i++) {
        		k = plan->index_tri[i];
        	    dx = plan->x_t[3 * index]-plan->x_s_tri[3 * i];
        	    dy = plan->x_t[3 * index + 1]-plan->x_s_tri[3 * i + 1];
        	    dz = plan->x_t[3 * index + 2]-plan->x_s_tri[3 * i + 2];

        	    R = dx * dx + dy * dy + dz*dz;

        	    dx *= plan->t_normal[3 * k];
        	    dy *= plan->t_normal[3 * k + 1];
        	    dz *= plan->t_normal[3 * k + 2];

        	    res += plan->charge_density[k]*(dx + dy + dz) / (R*sqrt(R));

        	}

        	return res;
        }
    }

}

fastsum_plan * create_plan() {

    fastsum_plan *plan = (fastsum_plan*) malloc(sizeof (fastsum_plan));
    plan->tree = NULL;

    return plan;
}

void init_fastsum(fastsum_plan *plan, int N_target, int triangle_num, int p, double mac, int num_limit, double correct_factor) {

    int i;

    plan->N_target = N_target;
    plan->N_source = triangle_num;
    
    plan->x_s = (double *) malloc(3 * plan->N_source * (sizeof (double)));

    plan->triangle_num = triangle_num;
    plan->x_s_tri = (double *) malloc(3 * plan->triangle_num * (sizeof (double)));

    plan->charge_density = (double *) malloc(plan->N_source * (sizeof (double)));
    plan->weights = (double *) malloc(plan->N_source * (sizeof (double)));

    plan->x_t = (double *) malloc(3 * N_target * (sizeof (double)));

    plan->triangle_nodes = (int *) malloc(3 * triangle_num * (sizeof (int)));
    plan->t_normal = (double *) malloc(3 * triangle_num * (sizeof (double)));

    plan->p = p;
    plan->mac_square = mac*mac;

    plan->index_tri = (int *) malloc(plan->triangle_num * (sizeof (int)));

    for (i = 0; i < triangle_num; i++) {
        plan->index_tri[i] = i;
    }

    plan->num_limit = num_limit;

    plan->vert_bsa = (double *) malloc(N_target * (sizeof (double)));
    plan->g2b = (int *) malloc(N_target * (sizeof (int)));

    plan->id_tn = (int *) malloc(N_target * (sizeof (int)));
    plan->id_nn = (int *) malloc(N_target * (sizeof (int)));

    plan->r_eps_factor=correct_factor;

    printf("N_source=%d  N_target=%d\n",plan->N_source,N_target);

}

void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal, int *triangle_nodes, int *g2b, double *vert_bsa) {

    int k;

    for (k = 0; k < 3 * plan->N_target; k++) {
        plan->x_t[k] = x_t[k];
    }

    for (k = 0; k < 3 * plan->triangle_num; k++) {
        plan->t_normal[k] = t_normal[k];
    }

    for (k = 0; k < 3 * plan->triangle_num; k++) {
        plan->triangle_nodes[k] = triangle_nodes[k];
    }

    for (k = 0; k < plan->N_target; k++) {
        plan->g2b[k] = g2b[k];
        plan->vert_bsa[k] = vert_bsa[k];
    }
}

void update_potential_u1(fastsum_plan *plan, double *u1) {

    int face, f, k;
    int i1, i2, i3;
    double sa, sb, sc;
    double tmp = 0;
    int index=0;

    for (face = 0; face < plan->triangle_num; face++) {

        f = 3 * face;

        i1 = plan->triangle_nodes[f];
        i2 = plan->triangle_nodes[f + 1];
        i3 = plan->triangle_nodes[f + 2];
        sa = u1[i1];
        sb = u1[i2];
        sc = u1[i3];

        tmp = sa + (sb - sa) / 3.0 + (sc - sa) / 3.0;
        plan->charge_density[face] = tmp * plan->weights[face];
        index++;

    }

}

void fastsum(fastsum_plan *plan, double *phi, double *u1) {
    int i, j, k;

    double ***a = alloc_3d_double(plan->p + 1, plan->p + 1, plan->p + 1);

    for (j = 0; j < plan->N_target; j++) {
         if (plan->g2b[j] >= 0) {
             phi[j] = compute_potential_single_target(plan, plan->tree, j, a);
         }
     }

     free_3d_double(a, plan->p + 1);

     compute_correction(plan,phi,u1);

}


void direct_sum(fastsum_plan *plan, double *phi, double *u1) {
    int i, j, k;
    double dx,dy,dz;
    double res,R;

    for (j = 0; j < plan->N_target; j++) {
    	 res=0;
    	 phi[j]=0;

         if (plan->g2b[j] >= 0) {

             for(i=0;i<plan->N_source;i++){

            	  k = plan->index_tri[i];
            	  dx = plan->x_t[3 * j]-plan->x_s_tri[3 * i];
            	  dy = plan->x_t[3 * j + 1]-plan->x_s_tri[3 * i + 1];
            	  dz = plan->x_t[3 * j + 2]-plan->x_s_tri[3 * i + 2];

            	  R = dx * dx + dy * dy + dz*dz;

            	  dx *= plan->t_normal[3 * k];
            	  dy *= plan->t_normal[3 * k + 1];
            	  dz *= plan->t_normal[3 * k + 2];

            	  res += plan->charge_density[k]*(dx + dy + dz) / (R*sqrt(R));
              }
              phi[j]=res;
         }
    }

    compute_correction(plan,phi,u1);

}

void compute_correction(fastsum_plan *plan, double *phi, double *u1) {
    int i, j, k,t;
    double dx,dy,dz;
    double R;
    int total_j = 0;
    int total_t = 0;

    for (i = 0; i < plan->N_target; i++) {
    	 if (plan->g2b[i] < 0) {
    	       continue;
    	 }

        for (j = 0; j < plan->id_tn[i]; j++) {

        	k=plan->id_t[total_t];

            dx = plan->x_t[3 * i]-plan->x_s_tri[3 * k];
            dy = plan->x_t[3 * i + 1]-plan->x_s_tri[3 * k + 1];
            dz = plan->x_t[3 * i + 2]-plan->x_s_tri[3 * k + 2];

            R = dx * dx + dy * dy + dz * dz;

            t=plan->index_tri[k];

            dx *= plan->t_normal[3 * t];
      	    dy *= plan->t_normal[3 * t + 1];
      	    dz *= plan->t_normal[3 * t + 2];

      	    phi[i] -= plan->charge_density[t]*(dx + dy + dz) / (R*sqrt(R));

      	    total_t++;
        }

        //printf("i=%d phi=   %g\n", i, phi[i]);


        for (j = 0; j < plan->id_nn[i]; j++) {
            k = plan->id_n[total_j];
            phi[i] += plan->b_m[total_j] * u1[k];
            total_j++;
        }

        phi[i]+=plan->vert_bsa[i] * u1[i];

    }

}


void copy_B(fastsum_plan *plan, double *B, int n) {
    int i;
    for (i = 0; i < n; i++) {
        B[i] = plan->b_m[i];
    }

}

void fastsum_finalize(fastsum_plan * plan) {

    free_tree(plan, plan->tree);

    free(plan->charge_density);
    free(plan->index_tri);
    free(plan->t_normal);
    free(plan->triangle_nodes);
    free(plan->weights);
    free(plan->x_s);
    free(plan->x_s_tri);
    free(plan->x_t);

    free(plan);

}

int main() {
    /**
    int p = 4;
    double mac = 0.5;
    int num_limit = 2;
    int i, j, k;
    int N_source = 8, N_target = 1;
    double res = 0, r;
    
    double ***a = alloc_3d_double(p + 1, p + 1, p + 1);

    compute_Taylor(a, 1, 2, 3, p);

    for (i = 0; i < p + 1; i++)
        for (j = 0; j < p + 1; j++)
            for (k = 0; k < p + 1; k++)
                printf("%d  %d  %d = %0.15f\n", i, j, k, a[i][j][k]);
     


    double xs[24]
            = {0.2, 0.7, 0,
        0.65, 0.9, 0,
        0.6, 0.8, 0,
        0.8, 0.7, 0,
        0.49, 0.49, 0,
        0.2, 0.1, 0,
        0.4, 0.2, 0,
        0.7, 0.2, 0};
    double xt[3] = {0, 0, 0};
    double density[8] = {1, 2, 3, 4, 5, 6, 7, 8};
     */

    double dx[3] = {1, 0, 0};
    double dy[3] = {0, 1, 0};
    double dz[3] = {0, 0, 0};

    printf("det2=%g\n", det2(dx, dy, dz));

    return 0;

}
