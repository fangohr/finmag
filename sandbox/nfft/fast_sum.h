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
    
    double ***moment;
    struct octree_node *children[8];
    
      
};

typedef struct {
    int N_source; //Number of the nodes with known charge density
    int N_target; //Number of the nodes to be evaluated 

    double *charge_density; // the coefficients of the source           

    double *x_s; //the coordinates of source nodes
    double *x_t; //the coordinates of target nodes
    
    double *x_s_bak; //the coordinates of source nodes in the original order
    
    int *index;
    
    int surface_n;
    int volume_n;
    double *s_x;
    double *s_y;
    double *s_w;
        
    double *t_normal;//store the normal of the triangles in the boundary
    int *face_nodes;//store the mapping between face and nodes
    int num_faces;

    double critical_sigma;
    struct octree_node *tree;
    int p;
    double mac_square;
    int num_limit;
    
} fastsum_plan;

fastsum_plan *create_plan();
void init_mesh(fastsum_plan *plan, double *x_s, double *x_t, double *t_normal, int *face_nodes);
void update_charge_density(fastsum_plan *plan, double *m,double *weight);
void fastsum_finalize(fastsum_plan *plan);
void fastsum_exact(fastsum_plan *plan, double *phi);
void fastsum(fastsum_plan *plan, double *phi);
void build_tree(fastsum_plan *plan);
void init_fastsum(fastsum_plan *plan, int N_source, int N_target, int surface_n,
        int volume_n, int num_faces, int p, double mac, int num_limit);

void compute_correction(fastsum_plan *plan, double *m, double *phi);



#endif	/* FAST_SUM_H */

