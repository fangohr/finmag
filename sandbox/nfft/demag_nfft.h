#ifndef DEMAG_NFFT_H
#define	DEMAG_NFFT_H


//#include "config.h"


#include <complex.h>

#include "nfft3util.h"
#include "nfft3.h"



#ifdef	__cplusplus
extern "C" {
#endif

    typedef struct {
        int N_source; //Number of the nodes with known charge density
        int N_target; //Number of the nodes to be evaluated 

        double _Complex *alpha; // the coefficients of the source           
        double _Complex *f; /**< target evaluations              */

        double *x; //the coordinates of source nodes, radius should smaller than 1/4-eps_b/2
        double *y; //the coordinates of target nodes


        int n; /**< expansion degree                */
        int n_total; /**< equals n*n*n for 3d case        */
        fftw_complex *b; /**< expansion coefficients          */

        int p; /**< degree of smoothness of regularization */
        double eps_I; /**< inner boundary                  */ /* fixed to p/n so far  */
        double eps_B; /**< outer boundary                  */ /* fixed to 1/16 so far */

        nfft_plan mv1; /**< source nfft plan                */
        nfft_plan mv2; /**< target nfft plan                */

        fftw_plan fft_plan;

        int Ad; /**< number of spline knots for nearfield computation of regularized kernel */
        double _Complex *Add; /**< spline values */


    } fastsum_plan;



    void fastsum_init_guru(fastsum_plan *plan, int N_total, int M_total, int nn, int m, int p, double eps_I, double eps_B);
    void fastsum_trafo(fastsum_plan *plan);
    void fastsum_precompute(fastsum_plan *plan);
    fastsum_plan* create_plan();
    void init_mesh(fastsum_plan *plan, double *x_s, double *x_t);
    void fastsum_finalize(fastsum_plan *plan);
    void update_charge(fastsum_plan *plan, double *charge);
    void get_phi(fastsum_plan *plan, double *phi);
    void fastsum_exact(fastsum_plan *plan);
    void fastsum_finalize(fastsum_plan *plan);


#ifdef	__cplusplus
}
#endif

#endif	/* DEMAG_NFFT_H */

