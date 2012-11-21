#include "fast_sum.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static const double dunavant_x[4][6] = {
    {0.333333333333333, 0, 0, 0, 0, 0},
    {0.666666666666667, 0.166666666666667, 0.166666666666667, 0, 0, 0},
    {0.333333333333333, 0.600000000000000, 0.200000000000000, 0.200000000000000, 0, 0},
    {0.108103018168070, 0.445948490915965, 0.445948490915965, 0.816847572980459, 0.091576213509771, 0.091576213509771}
};

static const double dunavant_y[4][6] = {
    {0.333333333333333, 0, 0, 0, 0, 0},
    {0.166666666666667, 0.166666666666667, 0.666666666666667, 0, 0, 0},
    {0.333333333333333, 0.200000000000000, 0.200000000000000, 0.600000000000000, 0, 0},
    {0.445948490915965, 0.445948490915965, 0.108103018168070, 0.091576213509771, 0.091576213509771, 0.816847572980459}
};

static const double dunavant_w[4][6] = {
    {1, 0, 0, 0, 0, 0},
    {0.333333333333333, 0.333333333333333, 0.333333333333333, 0, 0, 0},
    {-0.562500000000000, 0.520833333333333, 0.520833333333333, 0.520833333333333, 0, 0},
    {0.223381589678011, 0.223381589678011, 0.223381589678011, 0.109951743655322, 0.109951743655322, 0.109951743655322}
};


static const int dunavant_n[4] = {1, 3, 4, 6};

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

inline double G(double *x, double *y) {
    double r = 0;
    r = (x[0] - y[0])*(x[0] - y[0])
            +(x[1] - y[1])*(x[1] - y[1])
            +(x[2] - y[2])*(x[2] - y[2]);

    if (r <= 0) {
        return 0;
    }

    return 1.0 / sqrt(r);
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

void compute_source_nodes_weights(fastsum_plan *plan) {
    int face, f, k, tet, i, j;
    double x0, y0, z0;
    double x1, y1, z1;
    double x2, y2, z2;
    double x3, y3, z3;
    double v, *p, *p1, *p2, *p3;

    int sn = plan->triangle_p;
    int tri_n = dunavant_n[sn];


    int index = 0;
    int index_t = 0;

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

        for (k = 0; k < tri_n; k++) {
            plan->x_s[3 * index] = x2 * dunavant_x[sn][k] + x3 * dunavant_y[sn][k] + x1;
            plan->x_s[3 * index + 1] = y2 * dunavant_x[sn][k] + y3 * dunavant_y[sn][k] + y1;
            plan->x_s[3 * index + 2] = z2 * dunavant_x[sn][k] + z3 * dunavant_y[sn][k] + z1;

            plan->weights[index] = v * dunavant_w[sn][k] / 2.0;

            index++;
        }

    }

    for (k = 0; k < 3 * plan->N_source; k++) {
        plan->x_s_bak[k] = plan->x_s[k];
    }

}

inline double compute_correction_triangle(double *dx, double *dy, double *dz, double sa, double sb, double sc) {
    double r1, r2, r3;
    double fa, fb, fc, fd;
    r1 = distance(dx, dy);
    r2 = distance(dx, dz);
    r3 = distance(dy, dz);

    fa = det2(dx, dy, dz);

    fb = (sb - sc)*(r1 - r2) / (2 * r3 * r3);
    fc = (sb + sc + 2 * sa) / (4.0 * r3)+(r2 * r2 - r1 * r1)*(sb - sc) / (4.0 * r3 * r3 * r3);
    fd = log(r1 + r2 + r3) - log(r1 + r2 - r3);

    return fa * (fb + fc * fd);
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

        quicksort(0, plan->x_s + 3 * tree->begin, plan->index + tree->begin, tree->num_particle);

        for (k = tree->begin; k < tree->end; k++) {
            if (plan->x_s[3 * k] >= bnds[0][1])
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

            quicksort(1, plan->x_s + 3 * bend[i][0], plan->index + bend[i][0], bend[i][1] - bend[i][0]);

            for (k = bend[i][0]; k < bend[i][1]; k++) {
                if (plan->x_s[3 * k + 1] >= bnds[i][3])
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

            quicksort(2, plan->x_s + 3 * bend[i][0], plan->index + bend[i][0], bend[i][1] - bend[i][0]);

            for (k = bend[i][0]; k < bend[i][1]; k++) {
                if (plan->x_s[3 * k + 2] >= bnds[i][5])
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

void free_tree(fastsum_plan *plan,struct octree_node *tree) {
    int i;
    if (tree==NULL){
	return;
    }
    if (tree->num_children > 0) {
        for (i = 0; i < tree->num_children; i++) {
            free_tree(plan,tree->children[i]);
        }
    } else {
        free_3d_double(tree->moment,plan->p+1);
        free(tree);
    }

}

void build_tree(fastsum_plan *plan) {

    double bnd[6];

    plan->tree = (struct octree_node *) malloc(sizeof (struct octree_node));

    bnd[0] = array_min(plan->x_s, plan->N_source, 0);
    bnd[1] = array_max(plan->x_s, plan->N_source, 0);

    bnd[2] = array_min(plan->x_s, plan->N_source, 1);
    bnd[3] = array_max(plan->x_s, plan->N_source, 1);

    bnd[4] = array_min(plan->x_s, plan->N_source, 2);
    bnd[5] = array_max(plan->x_s, plan->N_source, 2);


    //printf("start to build tree  %d\n", plan->N_source);
    create_tree(plan, plan->tree, 0, plan->N_source, bnd);

    //printf("build tree okay\n");

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
    int i, j, k, ti, tj, tk;
    double tmp_x, tmp_y, tmp_z;

    int tri_n = dunavant_n[plan->triangle_p];
    //double tmp_moment = 0;

    for (i = 0; i < plan->p + 1; i++) {
        for (j = 0; j < plan->p - i + 1; j++) {
            for (k = 0; k < plan->p - i -j + 1; k++) {
                moment[i][j][k] = 0;
            }
        }
    }

    for (ti = tree->begin; ti < tree->end; ti++) {

        dx = plan->x_s[3 * ti] - x;
        dy = plan->x_s[3 * ti + 1] - y;
        dz = plan->x_s[3 * ti + 2] - z;
        tj = plan->index[ti];
	tk = tj/tri_n;

	dx *= plan->t_normal[3*tk];
	dy *= plan->t_normal[3*tk+1];	
	dz *= plan->t_normal[3*tk+2];

        tmp_x = 1.0;
        for (i = 0; i < plan->p + 1; i++) {
            tmp_y = 1.0;
            for (j = 0; j < plan->p - i + 1; j++) {
                tmp_z = 1.0;
                for (k = 0; k < plan->p - i - j + 1; k++) {
                    moment[i][j][k] += plan->charge_density[tj] * tmp_x * tmp_y * tmp_z;
                    tmp_z *= dz*(k+1);
                }
                tmp_y *= dy*(j+1);
            }
            tmp_x *= dx*(i+1);
        }

    }
}

double compute_potential_single_target(fastsum_plan *plan, struct octree_node *tree, int index, double ***a) {

    double R, rc;
    int i, j, k;
    double res = 0;
    double dx, dy, dz;
    int tri_n;
   

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

        res = 0;
        if (tree->num_children > 0) {

            for (i = 0; i < tree->num_children; i++) {
                res += compute_potential_single_target(plan, tree->children[i], index, a);
            }

            return res;

        } else {

	    tri_n = dunavant_n[plan->triangle_p];

            for (i = tree->begin; i < tree->end; i++) {

                j = plan->index[i];
                dx = plan->x_s[3 * i] - plan->x_t[3 * index];
                dy = plan->x_s[3 * i + 1] - plan->x_t[3 * index + 1];
                dz = plan->x_s[3 * i + 2] - plan->x_t[3 * index + 2];
                R = dx * dx + dy * dy + dz*dz;

		k = j/tri_n;
		dx *= plan->t_normal[3*k];
		dy *= plan->t_normal[3*k+1];	
		dz *= plan->t_normal[3*k+2];
		printf("j=%d  k=%d",j,k);

                if (R > 0) {
                    res += plan->charge_density[j]*(dx+dy+dz)/R;
                }
            }


            return res;
        }
    }


}

fastsum_plan *create_plan() {

    fastsum_plan *plan = (fastsum_plan*) malloc(sizeof (fastsum_plan));
    plan->tree=NULL;

    return plan;
}

void init_fastsum(fastsum_plan *plan, int N_target, int triangle_p,
       int triangle_num,  int p, double mac, int num_limit) {

    int i;

    plan->N_target = N_target;
    plan->N_source = triangle_num * dunavant_n[triangle_p];

    plan->x_s = (double *) malloc(3 * plan->N_source * (sizeof (double)));
    plan->x_s_bak = (double *) malloc(3 * plan->N_source * (sizeof (double)));
    plan->charge_density = (double *) malloc(plan->N_source * (sizeof (double)));
    plan->weights = (double *) malloc(plan->N_source * (sizeof (double)));


    plan->x_t = (double *) malloc(3 * N_target * (sizeof (double)));

    plan->triangle_p = triangle_p;
    plan->triangle_num = triangle_num;

    plan->triangle_nodes = (int *) malloc(3 * triangle_num * (sizeof (int)));
    plan->t_normal = (double *) malloc(3 * triangle_num * (sizeof (double)));

    plan->p = p;
    plan->mac_square = mac*mac;

    plan->index = (int *) malloc(plan->N_source * (sizeof (int)));

    for (i = 0; i < plan->N_source; i++) {
        plan->index[i] = i;
    }

    plan->num_limit = num_limit;


}

void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal, int *triangle_nodes) {

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
}

void update_potential_u1(fastsum_plan *plan, double *u1) {

    int face, f, k, tet, j;
    int i1, i2, i3, i4;
    double sa, sb, sc;


    double x1, y1, z1;
    double v, *p;
    double tmp = 0;


    int sn = plan->triangle_p;
    int n = dunavant_n[sn];
 
    int nt = plan->N_target;
    int index = 0;

    for (k = 0; k < plan->N_source; k++) {
        plan->charge_density[k] = 0;
    }


    for (face = 0; face < plan->triangle_num; face++) {

        f = 3 * face;

        i1 = plan->triangle_nodes[f];
	i2 = plan->triangle_nodes[f + 1];
        i3 = plan->triangle_nodes[f + 2];
        
        sa = u1[i1];
	sb = u1[i2];
	sc = u1[i3];

        for (k = 0; k < n; k++) {
            plan->charge_density[index++] = sa + (sb - sa) * dunavant_x[sn][k]
                    +(sc - sa) * dunavant_y[sn][k];
        }

    }


    for (k = 0; k < plan->N_source; k++) {
        plan->charge_density[k] *= plan->weights[k];
    }

}

inline double correction_over_triangle(fastsum_plan *plan, int base_index,
        double *dx, double *dy, double *dz, double sa, double sb, double sc) {

    double res = 0, exact = 0;

    int i, j;
    int n = dunavant_n[plan->triangle_p];

    for (i = 0; i < n; i++) {
        j = base_index + i;
        res += G(dx, &plan->x_s_bak[3 * j]) * plan->charge_density[j];
    }

    exact = compute_correction_triangle(dx, dy, dz, sa, sb, sc);

    return exact - res;
}



void fastsum(fastsum_plan *plan, double *phi) {
    int j = 0;

    //#pragma omp parallel
    {
       double ***a = alloc_3d_double(plan->p + 1, plan->p + 1, plan->p + 1);
   
       //#pragma omp for
       for (j = 0; j < plan->N_target; j++) {
         phi[j] = compute_potential_single_target(plan, plan->tree, j, a);
       }
       free_3d_double(a,plan->p+1);
    }

    

}

void fastsum_finalize(fastsum_plan *plan) {
    
   
    free_tree(plan,plan->tree);
    
    //free(plan->charge_density);
    //free(plan->index);
    //free(plan->t_normal);
    //free(plan->triangle_nodes);
    //free(plan->weights);
    //free(plan->x_s);
    //free(plan->x_s_bak);
    //free(plan->x_t);

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

    return 0;

}
