
#include<math.h>

typedef struct {
	int nxyz;

	double dt;
	double Ms;
	double c;
    double Q;
    double gamma;

	double theta;
	double theta1;
	double theta2;

	double *dm1;
	double *dm2;
	double *eta;

	int *pin;
	double *T;
	double *alpha;
	double *V;

} ode_solver;


ode_solver *create_ode_plan();
void init_solver(ode_solver *s,double *alpha, double *T, double *V,int nxyz);
void init_solver_parameters(ode_solver *s, double gamma, double Ms, double dt, double c);
void finalize_ode_plan(ode_solver *plan);
void run_step1(ode_solver *s, double *m, double *h, double *m_pred);
void run_step2(ode_solver *s, double *m_pred, double *h, double *m);


