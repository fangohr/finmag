#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct {
    double x,y,z;
} Cartesian_xyz;

struct octree_node {
    int num_children;
    int num_particle;

    int have_moment;
    int need_upadte_moment;

    int begin;
    int end;

    double x,y,z;//node x,y,z
    double rx,ry,rz;
    double radius_square;
    double radius;

    double ***moment;
    double *mom;
    struct octree_node *children[8];

};

inline double pow2(double x);
inline double det2(double *dx, double *dy, double *dz);
double **alloc_2d_double(int ndim1, int ndim2);
void free_2d_double(double **p);
double ***alloc_3d_double(int ndim1, int ndim2, int ndim3);
void free_3d_double(double ***p, int ndim1);




typedef struct {
    int N_source; //Number of the nodes with known charge density
    int N_target; //Number of the nodes to be evaluated

    double *charge_density; // the coefficients of the source
    double *weights;

    double *x_s; //the coordinates of source nodes
    double *x_t; //the coordinates of target nodes

    int *x_s_ids;

    int triangle_num;
    double *t_normal;//store the normal of the triangles in the boundary
    int *triangle_nodes;//store the mapping between face and nodes

    double critical_sigma;
    struct octree_node *tree;
    int p;
    double mac;
    double mac_square;
    int num_limit;

    double *vert_bsa;


    double r_eps;
    double r_eps_factor;
    double r_eps_squre;

    int *id_n; // indices nodes
    double *b_m;//boundary matrix
    int *id_nn;

    int total_length_n;
} fastsum_plan;

void compute_coefficient(double ***a, double dx, double dy, double dz, int p);
void compute_moment(fastsum_plan *plan, struct octree_node *tree, double ***moment, double x, double y, double z);
void reset_moment(fastsum_plan *plan, struct octree_node *tree);

fastsum_plan *create_plan(void);
void update_potential_u1(fastsum_plan *plan,double *u1);
void fastsum_finalize(fastsum_plan *plan);

void init_fastsum(fastsum_plan *plan, int N_target, int triangle_num, int p, double mac, int num_limit, double correct_factor);
void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal, int *triangle_nodes, double *vert_bsa);
void build_tree(fastsum_plan *plan);
void bulid_indices_I(fastsum_plan *plan);
void bulid_indices_II(fastsum_plan *plan);

void fast_sum_I(fastsum_plan *plan, double *phi,double *u1);
void fast_sum_II(fastsum_plan *plan, double *phi,double *u1);

void compute_source_nodes_weights(fastsum_plan *plan);

void direct_sum_I(fastsum_plan *plan, double *phi, double *u1);

double solid_angle_single(double *p, double *x1, double *x2, double *x3);
void boundary_element(double *xp, double *x1, double *x2, double *x3, double *res, double *T);
int get_total_length(fastsum_plan *plan);


void compute_coefficient_directly(double *a, double x, double y, double z, int p);
void compute_moment_directly(fastsum_plan *plan, struct octree_node *tree, double *moment, double x, double y, double z);

