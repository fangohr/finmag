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

static const double tet_x_nodes[2][4] = {
    {0.25, 0, 0, 0},
    {0.1381966011250110, 0.5854101966249680, 0.1381966011250110, 0.1381966011250110}
};

static const double tet_y_nodes[2][4] = {
    {0.25, 0, 0, 0},
    {0.1381966011250110, 0.1381966011250110, 0.5854101966249680, 0.1381966011250110}
};

static const double tet_z_nodes[2][4] = {
    {0.25, 0, 0, 0},
    {0.1381966011250110, 0.1381966011250110, 0.1381966011250110, 0.5854101966249680}
};

static const double tet_weights[2][4] = {
    {1.0, 0, 0, 0},
    {0.25, 0.25, 0.25, 0.25}
};

static const double tet_x_node3[35] = {
    0.0267367755543735, 0.9197896733368800, 0.0267367755543735, 0.0267367755543735,
    0.7477598884818090, 0.1740356302468940, 0.0391022406356488, 0.0391022406356488,
    0.0391022406356488, 0.0391022406356488, 0.1740356302468940, 0.7477598884818090,
    0.1740356302468940, 0.7477598884818090, 0.0391022406356488, 0.0391022406356488,
    0.4547545999844830, 0.0452454000155172, 0.0452454000155172, 0.4547545999844830,
    0.4547545999844830, 0.0452454000155172, 0.2232010379623150, 0.5031186450145980,
    0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150,
    0.0504792790607720, 0.0504792790607720, 0.0504792790607720, 0.5031186450145980,
    0.2232010379623150, 0.2232010379623150, 0.2500000000000000
};


static const double tet_y_node3[35] = {
    0.0267367755543735, 0.0267367755543735, 0.9197896733368800, 0.0267367755543735,
    0.0391022406356488, 0.0391022406356488, 0.7477598884818090, 0.1740356302468940,
    0.0391022406356488, 0.0391022406356488, 0.7477598884818090, 0.1740356302468940,
    0.0391022406356488, 0.0391022406356488, 0.1740356302468940, 0.7477598884818090,
    0.0452454000155172, 0.4547545999844830, 0.0452454000155172, 0.4547545999844830,
    0.0452454000155172, 0.4547545999844830, 0.2232010379623150, 0.2232010379623150,
    0.5031186450145980, 0.0504792790607720, 0.0504792790607720, 0.0504792790607720,
    0.2232010379623150, 0.5031186450145980, 0.2232010379623150, 0.2232010379623150,
    0.5031186450145980, 0.2232010379623150, 0.2500000000000000
};


static const double tet_z_node3[35] = {
    0.0267367755543735, 0.0267367755543735, 0.0267367755543735, 0.9197896733368800,
    0.0391022406356488, 0.0391022406356488, 0.0391022406356488, 0.0391022406356488,
    0.7477598884818090, 0.1740356302468940, 0.0391022406356488, 0.0391022406356488,
    0.7477598884818090, 0.1740356302468940, 0.7477598884818090, 0.1740356302468940,
    0.0452454000155172, 0.0452454000155172, 0.4547545999844830, 0.0452454000155172,
    0.4547545999844830, 0.4547545999844830, 0.0504792790607720, 0.0504792790607720,
    0.0504792790607720, 0.2232010379623150, 0.2232010379623150, 0.5031186450145980,
    0.2232010379623150, 0.2232010379623150, 0.5031186450145980, 0.2232010379623150,
    0.2232010379623150, 0.5031186450145980, 0.2500000000000000
};

static const double tet_weight3[35] = {
    0.0021900463965388, 0.0021900463965388, 0.0021900463965388, 0.0021900463965388,
    0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665,
    0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665,
    0.0143395670177665, 0.0143395670177665, 0.0143395670177665, 0.0143395670177665,
    0.0250305395686746, 0.0250305395686746, 0.0250305395686746, 0.0250305395686746,
    0.0250305395686746, 0.0250305395686746, 0.0479839333057554, 0.0479839333057554,
    0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554,
    0.0479839333057554, 0.0479839333057554, 0.0479839333057554, 0.0479839333057554,
    0.0479839333057554, 0.0479839333057554, 0.0931745731195340
};

static const int dunavant_n[4] = {1, 3, 4, 6};
static const int tet_quad_n[3] = {1, 4, 10};

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
    int pn = plan->tetrahedron_p;
    int tet_n = tet_quad_n[pn];
    double x_s_tmp[3 * 35], tmp;

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


    for (tet = 0; tet < plan->tetrahedron_num; tet++) {

        f = 4 * tet;

        k = plan->tetrahedron_nodes[f];
        p = &plan->x_t[3 * k];
        x0 = p[0];
        y0 = p[1];
        z0 = p[2];


        k = plan->tetrahedron_nodes[f + 1];
        p = &plan->x_t[3 * k];
        x1 = p[0] - x0;
        y1 = p[1] - y0;
        z1 = p[2] - z0;


        k = plan->tetrahedron_nodes[f + 2];
        p = &plan->x_t[3 * k];
        x2 = p[0] - x0;
        y2 = p[1] - y0;
        z2 = p[2] - z0;

        k = plan->tetrahedron_nodes[f + 3];
        p = &plan->x_t[3 * k];
        x3 = p[0] - x0;
        y3 = p[1] - y0;
        z3 = p[2] - z0;


        v = x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2
                + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2*z3;

        for (k = 0; k < tet_n; k++) {
            plan->x_s[3 * index] = x1 * tet_x_nodes[pn][k] + x2 * tet_y_nodes[pn][k] + x3 * tet_z_nodes[pn][k] + x0;
            plan->x_s[3 * index + 1] = y1 * tet_x_nodes[pn][k] + y2 * tet_y_nodes[pn][k] + y3 * tet_z_nodes[pn][k] + y0;
            plan->x_s[3 * index + 2] = z1 * tet_x_nodes[pn][k] + z2 * tet_y_nodes[pn][k] + z3 * tet_z_nodes[pn][k] + z0;

            plan->weights[index] = fabs(v) * tet_weights[pn][k] / 6.0;

            index++;
        }

        index_t = 0;
        for (k = 0; k < 35; k++) {
            x_s_tmp[3 * index_t] = x1 * tet_x_node3[k] + x2 * tet_y_node3[k] + x3 * tet_z_node3[k] + x0;
            x_s_tmp[3 * index_t + 1] = y1 * tet_x_node3[k] + y2 * tet_y_node3[k] + y3 * tet_z_node3[k] + y0;
            x_s_tmp[3 * index_t + 2] = z1 * tet_x_node3[k] + z2 * tet_y_node3[k] + z3 * tet_z_node3[k] + z0;

            index_t++;
        }

        for (i = 0; i < 4; i++) {
            k = plan->tetrahedron_nodes[f + i];
            p = &plan->x_t[3 * k];

            tmp = 0;
            for (j = 0; j < 35; j++) {
                p2 = &x_s_tmp[3 * j];
                tmp += G(p, p2) * fabs(v) * tet_weight3[j] / 6.0;
            }

            plan->tetrahedron_correction[f + i] = tmp;
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
        if (tree->have_moment) {
            free_3d_double(tree->moment, plan->p + 1);
        }
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


    for (i = 2; i < p; i++) {
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


    for (i = 2; i < p - 1; i++) {
        /*
        a[1][1][i] = (dx * a[0][1][i] + ddy * a[1][0][i] + ddz * a[1][1][i - 1] - a[1][1][i - 2]) * R;
        a[1][i][1] = (dx * a[0][i][1] + ddy * a[1][i - 1][1] + ddz * a[1][i][0] - a[1][i - 2][1]) * R;
        a[i][1][1] = (dy * a[i][0][i] + ddx * a[i - 1][1][1] + ddz * a[i][1][0] - a[i - 2][1][1]) * R;
        */
        
        a[1][1][i] = (dx * a[0][1][i] + ddy * a[1][0][i] + ddz * a[1][1][i - 1]- a[1][1][i - 2]) * R;
        
        a[1][i][1] = (dz * a[1][i][0] + ddx * a[0][i][1] + ddy * a[1][i - 1][1]- a[1][i - 2][1]) * R;
        
        a[i][1][1] = (dy * a[i][0][1] + ddz * a[i][1][0] + ddx * a[i - 1][1][1] - a[i - 2][1][1]) * R;
    }

    for (i = 2; i < p - 2; i++) {
        for (j = 2; j < p - i; j++) {
            a[1][i][j] = (dx * a[0][i][j] + ddy * a[1][i - 1][j] + ddz * a[1][i][j - 1]- a[1][ i - 2][j] - a[1][i][j - 2]) * R;
            a[i][1][j] = (dy * a[i][0][j] + ddx * a[i - 1][1][j] + ddz * a[i][1][ j - 1]- a[i - 2][1][j] - a[i][1][j - 2]) * R;
            a[i][j][1] = (dz * a[i][j][0] + ddx * a[i - 1][j][1] + ddy * a[i][j - 1][1]- a[i - 2][j][1] - a[i][j - 2][1]) * R;
        }
    }


    for (i = 2; i < p - 3; i++) {
        for (j = 2; j < p - i - 1; j++) {
            for (k = 2; k < p - i - j + 1; k++) {

                a[i][j][k] = (ddx * cf2[i] * a[i - 1][j][ k] + ddy * a[i][j - 1][k]
                        + ddz * a[i][j][k - 1] - cf[i] * a[i - 2][j][k]
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
    int i, j, k, ti, tj;
    double tmp_x, tmp_y, tmp_z;
    //double tmp_moment = 0;

    for (i = 0; i < plan->p + 1; i++) {
        for (j = 0; j < plan->p + 1; j++) {
            for (k = 0; k < plan->p + 1; k++) {
                moment[i][j][k] = 0;
            }
        }
    }

    for (ti = tree->begin; ti < tree->end; ti++) {

        dx = plan->x_s[3 * ti] - x;
        dy = plan->x_s[3 * ti + 1] - y;
        dz = plan->x_s[3 * ti + 2] - z;
        tj = plan->index[ti];

        tmp_x = 1.0;
        for (i = 0; i < plan->p + 1; i++) {
            tmp_y = 1.0;
            for (j = 0; j < plan->p - i + 1; j++) {
                tmp_z = 1.0;
                for (k = 0; k < plan->p - i - j + 1; k++) {
                    moment[i][j][k] += plan->charge_density[tj] * tmp_x * tmp_y * tmp_z;
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
            //printf("index=%d  xyz= %g %g  %g\n", index, plan->x_t[3 * index], plan->x_t[3 * index + 1], plan->x_t[3 * index + 2]);
            //printf("R=%g   begin=%d  end=%d  xyz=%g  %g  %g\n", R, tree->begin, tree->end, tree->x, tree->y, tree->z);
        }


        dx = plan->x_t[3 * index] - tree->x;
        dy = plan->x_t[3 * index + 1] - tree->y;
        dz = plan->x_t[3 * index + 2] - tree->z;
        compute_Taylor(a, dx, dy, dz, plan->p);

        for (i = 0; i < plan->p + 1; i++) {
            for (j = 0; j < plan->p - i + 1; j++) {
                for (k = 0; k < plan->p - i - j + 1; k++) {
                    // printf("%d %d  %d  =      %g   %g\n", i, j, k, a[i][j][k], tree->moment[i][j][k]);
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


            for (i = tree->begin; i < tree->end; i++) {

                j = plan->index[i];
                dx = plan->x_s[3 * i] - plan->x_t[3 * index];
                dy = plan->x_s[3 * i + 1] - plan->x_t[3 * index + 1];
                dz = plan->x_s[3 * i + 2] - plan->x_t[3 * index + 2];
                R = dx * dx + dy * dy + dz*dz;

                if (R > 0) {
                    res += plan->charge_density[j] / sqrt(R);
                }
            }


            return res;
        }
    }


}

fastsum_plan *create_plan() {

    fastsum_plan *str = (fastsum_plan*) malloc(sizeof (fastsum_plan));
    str->tree = NULL;

    return str;
}

void init_fastsum(fastsum_plan *plan, int N_target, int triangle_p,
        int tetrahedron_p, int triangle_num, int tetrahedron_num, int p, double mac, int num_limit) {

    int i;

    plan->N_target = N_target;
    plan->N_source = triangle_num * dunavant_n[triangle_p]
            + tetrahedron_num * tet_quad_n[tetrahedron_p];

    plan->x_s = (double *) malloc(3 * plan->N_source * (sizeof (double)));
    plan->x_s_bak = (double *) malloc(3 * plan->N_source * (sizeof (double)));
    plan->charge_density = (double *) malloc(plan->N_source * (sizeof (double)));
    plan->weights = (double *) malloc(plan->N_source * (sizeof (double)));


    plan->x_t = (double *) malloc(3 * N_target * (sizeof (double)));

    plan->triangle_p = triangle_p;
    plan->tetrahedron_p = tetrahedron_p;

    plan->triangle_num = triangle_num;
    plan->tetrahedron_num = tetrahedron_num;

    plan->triangle_nodes = (int *) malloc(3 * triangle_num * (sizeof (int)));
    plan->t_normal = (double *) malloc(3 * triangle_num * (sizeof (double)));

    plan->tetrahedron_nodes = (int *) malloc(4 * tetrahedron_num * (sizeof (int)));
    plan->tetrahedron_correction = (double *) malloc(4 * tetrahedron_num * (sizeof (double)));
    plan->tet_charge_density = (double *) malloc(tetrahedron_num * (sizeof (double)));

    plan->p = p;
    plan->mac_square = mac*mac;

    plan->index = (int *) malloc(plan->N_source * (sizeof (int)));

    for (i = 0; i < plan->N_source; i++) {
        plan->index[i] = i;
    }

    plan->num_limit = num_limit;


}

void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal,
        int *triangle_nodes, int *tetrahedron_nodes) {

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

    for (k = 0; k < 4 * plan->tetrahedron_num; k++) {
        plan->tetrahedron_nodes[k] = tetrahedron_nodes[k];
    }

}

void update_charge_density(fastsum_plan *plan, double *m) {

    int face, f, k, tet, j;
    int i1, i2, i3;
    double sa, sb, sc;
    double m0[3], m1[3], m2[3], m3[3];
    double x0, y0, z0;
    double x1, y1, z1;
    double x2, y2, z2;
    double x3, y3, z3;
    double a1[3], a2[3], a3[3];
    double v, *p;
    double tmp = 0;


    int sn = plan->triangle_p;
    int n = dunavant_n[sn];
    int tet_n = tet_quad_n[plan->tetrahedron_p];

    int nt = plan->N_target;
    int index = 0;

    for (k = 0; k < plan->N_source; k++) {
        plan->charge_density[k] = 0;
    }


    for (face = 0; face < plan->triangle_num; face++) {

        f = 3 * face;

        i1 = plan->triangle_nodes[f];
        i2 = plan->N_target + i1;
        i3 = plan->N_target + i2;
        sa = (m[i1] * plan->t_normal[f] + m[i2] * plan->t_normal[f + 1] + m[i3] * plan->t_normal[f + 2]);


        i1 = plan->triangle_nodes[f + 1];
        i2 = plan->N_target + i1;
        i3 = plan->N_target + i2;
        sb = (m[i1] * plan->t_normal[f] + m[i2] * plan->t_normal[f + 1] + m[i3] * plan->t_normal[f + 2]);


        i1 = plan->triangle_nodes[f + 2];
        i2 = plan->N_target + i1;
        i3 = plan->N_target + i2;
        sc = (m[i1] * plan->t_normal[f] + m[i2] * plan->t_normal[f + 1] + m[i3] * plan->t_normal[f + 2]);


        for (k = 0; k < n; k++) {
            plan->charge_density[index++] = sa + (sb - sa) * dunavant_x[sn][k]
                    +(sc - sa) * dunavant_y[sn][k];
        }

    }


    for (tet = 0; tet < plan->tetrahedron_num; tet++) {

        f = 4 * tet;

        k = plan->tetrahedron_nodes[f];
        p = &plan->x_t[3 * k];
        x0 = p[0];
        y0 = p[1];
        z0 = p[2];
        m0[0] = m[k];
        m0[1] = m[nt + k];
        m0[2] = m[nt + nt + k];

        k = plan->tetrahedron_nodes[f + 1];
        p = &plan->x_t[3 * k];
        x1 = p[0] - x0;
        y1 = p[1] - y0;
        z1 = p[2] - z0;
        m1[0] = m[k] - m0[0];
        m1[1] = m[nt + k] - m0[1];
        m1[2] = m[nt + nt + k] - m0[2];

        k = plan->tetrahedron_nodes[f + 2];
        p = &plan->x_t[3 * k];
        x2 = p[0] - x0;
        y2 = p[1] - y0;
        z2 = p[2] - z0;
        m2[0] = m[k] - m0[0];
        m2[1] = m[nt + k] - m0[1];
        m2[2] = m[nt + nt + k] - m0[2];

        k = plan->tetrahedron_nodes[f + 3];
        p = &plan->x_t[3 * k];
        x3 = p[0] - x0;
        y3 = p[1] - y0;
        z3 = p[2] - z0;
        m3[0] = m[k] - m0[0];
        m3[1] = m[nt + k] - m0[1];
        m3[2] = m[nt + nt + k] - m0[2];

        a1[0] = y3 * z2 - y2*z3;
        a1[1] = -x3 * z2 + x2*z3;
        a1[2] = x3 * y2 - x2*y3;
        a2[0] = -y3 * z1 + y1*z3;
        a2[1] = x3 * z1 - x1*z3;
        a2[2] = -x3 * y1 + x1*y3;
        a3[0] = y2 * z1 - y1*z2;
        a3[1] = -x2 * z1 + x1*z2;
        a3[2] = x2 * y1 - x1*y2;

        v = x3 * y2 * z1 - x2 * y3 * z1 - x3 * y1 * z2
                + x1 * y3 * z2 + x2 * y1 * z3 - x1 * y2*z3;

        tmp = 0;
        for (j = 0; j < 3; j++) {
            tmp += a1[j] * m1[j] + a2[j] * m2[j] + a3[j] * m3[j];
        }

        tmp = -1.0 * tmp / v;

        for (j = 0; j < tet_n; j++) {
            plan->charge_density[index++] = tmp;
        }

        plan->tet_charge_density[tet] = tmp;

    }


    for (k = 0; k < plan->N_source; k++) {
        plan->charge_density[k] *= plan->weights[k];
        //printf("k=%d  %f\n",plan->weights[k]);
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

inline double correction_over_tetrahedron(fastsum_plan *plan, int base_index,
        int tet_index, int index_p, double *p) {

    double res = 0, exact = 0;

    int i, j;
    int n = tet_quad_n[plan->tetrahedron_p];


    for (i = 0; i < n; i++) {
        j = base_index + i;
        res += G(p, &plan->x_s_bak[3 * j]) * plan->charge_density[j];
    }

    exact = plan->tet_charge_density[tet_index] * plan->tetrahedron_correction[index_p];

    return exact - res;
}

void compute_correction(fastsum_plan *plan, double *m, double *phi) {

    int face, f;
    int i, j, k, i2, i3;
    int base_index = 0;
    int tet;
    double sa, sb, sc;
    double *x, *y, *z, *p;
    int sn = plan->triangle_p;
    int n = dunavant_n[sn];
    int tet_n = tet_quad_n[plan->tetrahedron_p];
    double tmp;

    for (face = 0; face < plan->triangle_num; face++) {

        f = 3 * face;

        i = plan->triangle_nodes[f];
        i2 = plan->N_target + i;
        i3 = plan->N_target + i2;
        x = &plan->x_t[3 * i];
        sa = (m[i] * plan->t_normal[f] + m[i2] * plan->t_normal[f + 1] + m[i3] * plan->t_normal[f + 2]);


        j = plan->triangle_nodes[f + 1];
        i2 = plan->N_target + j;
        i3 = plan->N_target + i2;
        y = &plan->x_t[3 * j];
        sb = (m[j] * plan->t_normal[f] + m[i2] * plan->t_normal[f + 1] + m[i3] * plan->t_normal[f + 2]);


        k = plan->triangle_nodes[f + 2];
        i2 = plan->N_target + k;
        i3 = plan->N_target + i2;
        z = &plan->x_t[3 * k];
        sc = (m[k] * plan->t_normal[f] + m[i2] * plan->t_normal[f + 1] + m[i3] * plan->t_normal[f + 2]);


        phi[i] += correction_over_triangle(plan, base_index, x, y, z, sa, sb, sc);
        phi[j] += correction_over_triangle(plan, base_index, y, z, x, sb, sc, sa);
        phi[k] += correction_over_triangle(plan, base_index, z, x, y, sc, sa, sb);
        base_index += n;

    }


    for (tet = 0; tet < plan->tetrahedron_num; tet++) {

        f = 4 * tet;

        for (i = 0; i < 4; i++) {
            k = plan->tetrahedron_nodes[f + i];
            p = &plan->x_t[3 * k];
            tmp = correction_over_tetrahedron(plan, base_index, tet, f + i, p);
            phi[k] += tmp;
            //printf("tmp=%f  phi[k]=%f\n",tmp,phi[k]);
        }

        base_index += tet_n;
    }

}

void fastsum_exact(fastsum_plan *plan, double *phi) {
    int i, j, k;
    double r;

    for (j = 0; j < plan->N_target; j++) {
        phi[j] = 0;
        for (k = 0; k < plan->N_source; k++) {
            r = pow2(plan->x_t[3 * j] - plan->x_s[3 * k])
                    + pow2(plan->x_t[3 * j + 1] - plan->x_s[3 * k + 1])
                    + pow2(plan->x_t[3 * j + 2] - plan->x_s[3 * k + 2]);
            if (r > 0) {
                i = plan->index[k];
                phi[j] += plan->charge_density[i] / sqrt(r);
            }
        }
    }
}

void fastsum(fastsum_plan *plan, double *phi) {
    int j = 0;

    double ***a;
    //#pragma omp parallel private(a)
    {
        a=alloc_3d_double(plan->p + 1, plan->p + 1, plan->p + 1);  
	//#pragma omp for
        for (j = 0; j < plan->N_target; j++) {
            phi[j] = compute_potential_single_target(plan, plan->tree, j, a);
        }
        free_3d_double(a, plan->p + 1);
    }



}

void fastsum_finalize(fastsum_plan *plan) {


    free_tree(plan, plan->tree);

    free(plan->charge_density);
    free(plan->index);
    free(plan->t_normal);
    free(plan->tet_charge_density);
    free(plan->tetrahedron_correction);
    free(plan->tetrahedron_nodes);
    free(plan->triangle_nodes);
    free(plan->weights);
    free(plan->x_s);
    free(plan->x_s_bak);
    free(plan->x_t);

    free(plan);

}

int main(void) {
    
    int p = 4;
    double mac = 0.5;
    int num_limit = 2;
    int i, j, k;
    int N_source = 8, N_target = 1;
    double res = 0;
    
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
    
    fastsum_plan *plan=create_plan();
    
    plan->N_source=N_source;
    plan->N_target=N_target;
    plan->num_limit=num_limit;
    plan->x_s = xs;
    plan->x_t = xt;
    plan->x_s_bak = (double *) malloc(3 * plan->N_source * (sizeof (double)));
    
    for(i=0;i<plan->N_source;i++){
        printf("%f\n",plan->x_s[i]);
    }
    
    plan->charge_density=density;
    
    plan->p = p;
    plan->mac_square = mac*mac;

    plan->index = (int *) malloc(plan->N_source * (sizeof (int)));

    for (i = 0; i < plan->N_source; i++) {
        plan->index[i] = i;
    }
    
    build_tree(plan);
    
    
    double exact[1];
    fastsum_exact(plan,exact);
    printf("%0.15f\n",exact[0]);
    
    for(p=2;p<10;p++){
        plan->p=p;
        build_tree(plan);
        a = alloc_3d_double(p + 1, p + 1, p + 1);
        res=compute_potential_single_target(plan, plan->tree, 0, a);
        printf("p=%d    %0.15f   rel_error=%e\n",p,res,(exact[0]-res)/exact[0]);
    }

    return 0;

}
