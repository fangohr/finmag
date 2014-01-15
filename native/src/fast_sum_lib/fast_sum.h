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
    double *weights;

    double *x_s; //the coordinates of source nodes
    double *x_t; //the coordinates of target nodes
    
    double *x_s_bak; //the coordinates of source nodes in the original order
    //double *x_s_tet;//the coordinates of source nodes used for tetrahedron correction 
    
    int *index;
    
    int triangle_p;
    int tetrahedron_p;
        
    int triangle_num;
    double *t_normal;//store the normal of the triangles in the boundary
    int *triangle_nodes;//store the mapping between face and nodes
    
    int tetrahedron_num;
    int *tetrahedron_nodes;//store the mapping between tetrahedron and nodes
    double *tetrahedron_correction;//store the correction coefficients 
    double *tet_charge_density;//used for  correction too

    double critical_sigma;
    struct octree_node *tree;
    int p;
    double mac_square;
    int num_limit;
    
} fastsum_plan;

fastsum_plan *create_plan(void);
void init_mesh(fastsum_plan *plan, double *x_t, double *t_normal,
        int *triangle_nodes, int *tetrahedron_nodes);
void update_charge_density(fastsum_plan *plan,double *m);
void fastsum_finalize(fastsum_plan *plan);
void fastsum_exact(fastsum_plan *plan, double *phi);
void fastsum(fastsum_plan *plan, double *phi);
void build_tree(fastsum_plan *plan);
void init_fastsum(fastsum_plan *plan, int N_target, int triangle_p,
        int tetrahedron_p, int triangle_num, int tetrahedron_num, int p, double mac, int num_limit);

void compute_correction(fastsum_plan *plan, double *m, double *phi);
void compute_source_nodes_weights(fastsum_plan *plan);

#endif	/* FAST_SUM_H */

