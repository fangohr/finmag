#ifndef FAST_SUM_H
#define	FAST_SUM_H


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
    struct octree_node *children[8];
    
      
};

typedef struct {
    int N_source; //Number of the nodes with known charge density
    int N_target; //Number of the nodes to be evaluated 

    double *charge_density; // the coefficients of the source 
    double *weights;

    double *x_s; //the coordinates of source nodes
    double *x_t; //the coordinates of target nodes
    
    double *x_s_tri; //a triangle as a source point, needed in the analytical correction

    
    int *x_s_ids;
    
        
    int triangle_num;
    double *t_normal;//store the normal of the triangles in the boundary
    int *triangle_nodes;//store the mapping between face and nodes
    
    double critical_sigma;
    struct octree_node *tree;
    int p;
    double mac_square;
    int num_limit;

    double *vert_bsa;

    double r_eps;

    int *id_n; // indices nodes
    double *b_m;//boundary matrix
    //int *id_tn;
    int *id_nn;
  int total_length_n;
} fastsum_plan;

fastsum_plan *create_plan();
void update_potential_u1(fastsum_plan *plan,double *u1);
void fastsum_finalize(fastsum_plan *plan);

void fastsum(fastsum_plan *plan, double *phi,double *u1);
void build_tree(fastsum_plan *plan);
void bulid_indices(fastsum_plan *plan);
void init_fastsum(fastsum_plan *plan, int N_target, int triangle_num, int p, double mac, int num_limit);
void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal, int *triangle_nodes, double *vert_bsa);

void compute_triangle_source_nodes(fastsum_plan *plan);
void compute_source_nodes_weights(fastsum_plan *plan);

double solid_angle_single(double *p, double *x1, double *x2, double *x3);
void copy_B(fastsum_plan *plan, double *B, int n);//used for test
void boundary_element(double *xp, double *x1, double *x2, double *x3, double *res);
int get_total_length(fastsum_plan *plan);
void print_tree(fastsum_plan *plan);
#endif	/* FAST_SUM_H */

