
#include "demag_nfft.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

double kernel(double x, int der) {
    double value = 0.0;


    if (fabs(x) < 1e-12) value = 0.0;
    else switch (der) {
            case 0: value = 1.0 / fabs(x);
                break;
            case 1: value = -1 / x / fabs(x);
                break;
            case 2: value = 2.0 / pow(fabs(x), 3.0);
                break;
            case 3: value = -6.0 / (x * x * x) / fabs(x);
                break;
            case 4: value = 24.0 / pow(fabs(x), 5.0);
                break;
            case 5: value = -120.0 / (x * x * x * x * x) / fabs(x);
                break;
            case 6: value = 720.0 / pow(fabs(x), 7.0);
                break;
            case 7: value = -5040.0 / (x * x * x * x * x * x * x) / fabs(x);
                break;
            case 8: value = 40320.0 / pow(fabs(x), 9.0);
                break;
            case 9: value = -362880.0 / (x * x * x * x * x * x * x * x * x) / fabs(x);
                break;
            case 10: value = 3628800.0 / pow(fabs(x), 11.0);
                break;
            case 11: value = -39916800.0 / pow(x, 11.0) / fabs(x);
                break;
            case 12: value = 479001600.0 / pow(fabs(x), 13.0);
                break;
            default: value = 0.0;
        }

    return value;
}

/** cubic spline interpolation in near field with even kernels */
double _Complex kubintkern(const double x, const double _Complex *Add,
  const int Ad, const double a)
{
  double c,c1,c2,c3,c4;
  int r;
  double _Complex f0,f1,f2,f3;
  c=x*Ad/a;
  r=c; r=abs(r);
  if (r==0) {f0=Add[r+1];f1=Add[r];f2=Add[r+1];f3=Add[r+2];}
  else { f0=Add[r-1];f1=Add[r];f2=Add[r+1];f3=Add[r+2];}
  c=fabs(c);
  c1=c-r;
  c2=c1+1.0;
  c3=c1-1.0;
  c4=c1-2.0;
  /* return(-f0*(c-r)*(c-r-1.0)*(c-r-2.0)/6.0+f1*(c-r+1.0)*(c-r-1.0)*(c-r-2.0)/2-
     f2*(c-r+1.0)*(c-r)*(c-r-2.0)/2+f3*(c-r+1.0)*(c-r)*(c-r-1.0)/6.0); */
  return(-f0*c1*c3*c4/6.0+f1*c2*c3*c4/2.0-f2*c2*c1*c4/2.0+f3*c2*c1*c3/6.0);
}

/** quicksort algorithm for source knots and associated coefficients */
void quicksort(int t, double *x, double _Complex *alpha, int N)
{
  int lpos=0;
  int rpos=N-1;
  
  double pivot=x[(N/2)*3+t];

  int k;
  double temp1;
  double _Complex temp2;

  while (lpos<=rpos)
  {
    while (x[3*lpos+t]<pivot)
      lpos++;
    while (x[3*rpos+t]>pivot)
      rpos--;
    if (lpos<=rpos)
    {
      for (k=0; k<3; k++)
      {
        temp1=x[3*lpos+k];
        x[3*lpos+k]=x[3*rpos+k];
        x[3*rpos+k]=temp1;
      }
      temp2=alpha[lpos];
      alpha[lpos]=alpha[rpos];
      alpha[rpos]=temp2;

      lpos++;
      rpos--;
    }
  }
  if (0<rpos)
    quicksort(t,x,alpha,rpos+1);
  if (lpos<N-1)
    quicksort(t,x+3*lpos,alpha+lpos,N-lpos);
}

/** recursive sort of source knots dimension by dimension to get tree structure */
void BuildTree(int t, double *x, double _Complex *alpha, int N)
{
  if (N>1)
  {
    int m=N/2;

    quicksort(t,x,alpha,N);

    BuildTree((t+1)%3, x, alpha, m);
    BuildTree((t+1)%3, x+(m+1)*3, alpha+(m+1), N-m-1);
  }
}



/** max */
int max_i(int a, int b) {
    return a >= b ? a : b;
}

/** factorial */
double fak(int n) {
    if (n <= 1) return 1.0;
    else return (double) n * fak(n - 1);
}

/** binomial coefficient */
double binom(int n, int m) {
    return fak(n) / fak(m) / fak(n - m);
}

/** basis polynomial for regularized kernel */
double BasisPoly(int m, int r, double xx) {
    int k;
    double sum = 0.0;

    for (k = 0; k <= m - r; k++) {
        sum += binom(m + k, k) * pow((xx + 1.0) / 2.0, (double) k);
    }
    return sum * pow((xx + 1.0), (double) r) * pow(1.0 - xx, (double) (m + 1)) / (1 << (m + 1)) / fak(r); /* 1<<(m+1) = 2^(m+1) */
}

/** regularized kernel with K_I arbitrary and K_B smooth to zero */
double _Complex regkern(double xx, int p, double a, double b) {
    int r;
    double _Complex sum = 0.0;

    if (xx<-0.5)
        xx = -0.5;
    if (xx > 0.5)
        xx = 0.5;
    if ((xx >= -0.5 + b && xx <= -a) || (xx >= a && xx <= 0.5 - b)) {
        return kernel(xx, 0);
    } else if (xx<-0.5 + b) {
        sum = (kernel(-0.5, 0) + kernel(0.5, 0)) / 2.0
                * BasisPoly(p - 1, 0, 2.0 * xx / b + (1.0 - b) / b);
        for (r = 0; r < p; r++) {
            sum += pow(-b / 2.0, (double) r)
                    * kernel(-0.5 + b, r)
                    * BasisPoly(p - 1, r, -2.0 * xx / b + (b - 1.0) / b);
        }
        return sum;
    } else if ((xx>-a) && (xx < a)) {
        for (r = 0; r < p; r++) {
            sum += pow(a, (double) r)
                    *(kernel(-a, r) * BasisPoly(p - 1, r, xx / a)
                    + kernel(a, r) * BasisPoly(p - 1, r, -xx / a)*(r & 1 ? -1 : 1));
        }
        return sum;
    } else if (xx > 0.5 - b) {
        sum = (kernel(-0.5, 0) + kernel(0.5, 0)) / 2.0
                * BasisPoly(p - 1, 0, -2.0 * xx / b + (1.0 - b) / b);
        for (r = 0; r < p; r++) {
            sum += pow(b / 2.0, (double) r)
                    * kernel(0.5 - b, r)
                    * BasisPoly(p - 1, r, 2.0 * xx / b - (1.0 - b) / b);
        }
        return sum;
    }
    return kernel(xx, 0);
}

/** regularized kernel for even kernels with K_I even
 *  and K_B mirrored smooth to K(1/2) (used in dD, d>1)
 */
double regkern3(double xx, int p, double a, double b) {
    int r;
    double _Complex sum = 0.0;

    xx = fabs(xx);

    if (xx >= 0.5) {
        /*return kern(typ,c,0,0.5);*/
        xx = 0.5;
    }
    /* else */
    if ((a <= xx) && (xx <= 0.5 - b)) {
        return kernel(xx, 0);
    } else if (xx < a) {
        for (r = 0; r < p; r++) {
            sum += pow(-a, (double) r) * kernel(a, r)
                    *(BasisPoly(p - 1, r, xx / a) + BasisPoly(p - 1, r, -xx / a));
        }
        /*sum=kern(typ,c,0,xx); */
        return sum;
    } else if ((0.5 - b < xx) && (xx <= 0.5)) {
        sum = kernel(0.5, 0) * BasisPoly(p - 1, 0, -2.0 * xx / b + (1.0 - b) / b);

        for (r = 0; r < p; r++) {
            sum += pow(b / 2.0, (double) r)
                    * kernel(0.5 - b, r)
                    * BasisPoly(p - 1, r, 2.0 * xx / b - (1.0 - b) / b);
        }
        return sum;
    }
    return 0.0;
}




/** fast search in tree of source knots for near field computation*/
double _Complex SearchTree(const int t, const double *x,
  const double _Complex *alpha, const double *xmin, const double *xmax,
  const int N,  const int Ad,
  const double _Complex *Add, const int p)
{
  int m=N/2;
  double Min=xmin[t], Max=xmax[t], Median=x[m*3+t];
  double a=fabs(Max-Min)/2;
  int l;
  int E=0;
  double r;

  if (N==0)
    return 0.0;
  if (Min>Median)
    return SearchTree((t+1)%3,x+(m+1)*3,alpha+(m+1),xmin,xmax,N-m-1,Ad,Add,p);
  else if (Max<Median)
    return SearchTree((t+1)%3,x,alpha,xmin,xmax,m,Ad,Add,p);
  else
  {
    double _Complex result = 0.0;
    E=0;

    for (l=0; l<3; l++)
    {
      if ( x[m*3+l]>xmin[l] && x[m*3+l]<xmax[l] )
        E++;
    }

    if (E==3)
    {
      
        r=0.0;
        for (l=0; l<3; l++)
          r+=(xmin[l]+a-x[m*3+l])*(xmin[l]+a-x[m*3+l]);  /* remember: xmin+a = y */
        r=sqrt(r);
      
      if (fabs(r)<a)
      {
        result += alpha[m]*kernel(r,0);                         /* alpha*(kern-regkern) */
       
        //result -= alpha[m]*regkern(r,p,a,1.0/16.0); 
        result -= alpha[m]*kubintkern(r,Add,Ad,a);                /* spline approximation */
      }
    }
    result += SearchTree((t+1)%3,x+(m+1)*3,alpha+(m+1),xmin,xmax,N-m-1,Ad,Add,p)
      + SearchTree((t+1)%3,x,alpha,xmin,xmax,m,Ad,Add,p);
    return result;
  }
}




fastsum_plan* create_plan() {

    fastsum_plan *str = (fastsum_plan*) malloc(sizeof (fastsum_plan));

    return str;
}

void fastsum_finalize(fastsum_plan *plan) {
    nfft_free(plan->x);
    nfft_free(plan->alpha);
    nfft_free(plan->y);
    nfft_free(plan->f);

    nfft_free(plan->Add);

    nfft_finalize(&(plan->mv1));
    nfft_finalize(&(plan->mv2));


    fftw_destroy_plan(plan->fft_plan);

    nfft_free(plan->b);

    free(plan);
}

void init_mesh(fastsum_plan *plan, double *x_s, double *x_t) {
   
  int k;

    for (k = 0; k < 3*plan->N_source; k++) {

        plan->x[k] = x_s[k];
        
    }

    for (k = 0; k < 3 * plan->N_target; k++) {
        
        plan->y[k] = x_t[k];
      
    }

    for (k = 0; k < 3* plan->N_target; k++) {
      //printf("y[%d]=%f  x=%f \n",k, plan->y[k],x_t[k]);
    }

    //printf("init_mesh okay!\n");
}

void update_charge(fastsum_plan *plan, double *charge) {

    int k;

    for (k = 0; k < plan->N_source; k++) {

        plan->alpha[k] = charge[k];

    }

    // printf("update charge density okay!\n");
}

void get_phi(fastsum_plan *plan, double *phi) {

    int k;

    for (k = 0; k < plan->N_target; k++) {

        phi[k] = plan->f[k];

    }

}

void fastsum_exact(fastsum_plan *plan) {
    int j, k;
    int t;
    double r;


    for (j = 0; j < plan->N_target; j++) {
        plan->f[j] = 0.0;
        for (k = 0; k < plan->N_source; k++) {

            r = 0.0;
            for (t = 0; t < 3; t++)
                r += (plan->y[j * 3 + t] - plan->x[k * 3 + t])*(plan->y[j * 3 + t] - plan->x[k * 3 + t]);
            r = sqrt(r);

            plan->f[j] += plan->alpha[k] * kernel(r, 0);
        }
    }
}

/** initialization of fastsum plan */
void fastsum_init_guru(fastsum_plan *plan, int N_total, int M_total, int nn, int m, int p, double eps_I, double eps_B) {

    int t;
    int N[3], n[3];


    int sort_flags_trafo = NFFT_SORT_NODES;
    int sort_flags_adjoint = NFFT_SORT_NODES;

    plan->N_source = N_total;
    plan->N_target = M_total;

    plan->x = (double *) nfft_malloc(3 * N_total * (sizeof (double)));
    plan->alpha = (double _Complex*) nfft_malloc(N_total * (sizeof (double _Complex)));

    plan->y = (double *) nfft_malloc(3 * M_total * (sizeof (double)));
    plan->f = (double _Complex*) nfft_malloc(M_total * (sizeof (double _Complex)));

    plan->n = nn;
    plan->n_total = nn * nn* nn;


    plan->p = p;
    plan->eps_I = eps_I; /* =(double)ths->p/(double)nn; */ /** inner boundary */
    plan->eps_B = eps_B; /* =1.0/16.0; */ /** outer boundary */


    for (t = 0; t < 3; t++) {
        N[t] = nn;
        n[t] = 2 * nn;
    }
    nfft_init_guru(&(plan->mv1), 3, N, N_total, n, m,
            sort_flags_adjoint |
            PRE_PHI_HUT | PRE_PSI | MALLOC_X | MALLOC_F_HAT | MALLOC_F | FFTW_INIT | FFT_OUT_OF_PLACE,
            FFTW_MEASURE | FFTW_DESTROY_INPUT);
    nfft_init_guru(&(plan->mv2), 3, N, M_total, n, m,
            sort_flags_trafo |
            PRE_PHI_HUT | PRE_PSI | MALLOC_X | MALLOC_F_HAT | MALLOC_F | FFTW_INIT | FFT_OUT_OF_PLACE,
            FFTW_MEASURE | FFTW_DESTROY_INPUT);


    plan->b = (fftw_complex *) nfft_malloc(plan->n_total * sizeof (fftw_complex));

    plan->fft_plan = fftw_plan_dft(3, N, plan->b, plan->b, FFTW_FORWARD, FFTW_ESTIMATE);

    plan->Ad = 2 * (plan->p)*(plan->p);
    plan->Add = (double _Complex *)nfft_malloc((plan->Ad + 3)*(sizeof (double _Complex)));

    //printf("init_guru okay!\n");
}

/** precomputation for fastsum */
void fastsum_precompute(fastsum_plan *plan) {
    int j, k, t;

    BuildTree(0,plan->x,plan->alpha,plan->N_source);
    
    for (k=0; k <= plan->Ad+2; k++)
        plan->Add[k] = regkern3(plan->eps_I*(double)k/plan->Ad, plan->p, plan->eps_I, plan->eps_B);


    /** init NFFT plan for transposed transform in first step*/
    for (k = 0; k < plan->mv1.M_total; k++)
        for (t = 0; t < plan->mv1.d; t++)
            plan->mv1.x[plan->mv1.d * k + t] = -plan->x[plan->mv1.d * k + t]; /* note the factor -1 for transposed transform instead of adjoint*/

    /** precompute psi, the entries of the matrix B */
    if (plan->mv1.nfft_flags & PRE_LIN_PSI)
        nfft_precompute_lin_psi(&(plan->mv1));

    if (plan->mv1.nfft_flags & PRE_PSI)
        nfft_precompute_psi(&(plan->mv1));

    if (plan->mv1.nfft_flags & PRE_FULL_PSI)
        nfft_precompute_full_psi(&(plan->mv1));


    /** init Fourier coefficients */
    for (k = 0; k < plan->mv1.M_total; k++)
        plan->mv1.f[k] = plan->alpha[k];

    /** init NFFT plan for transform in third step*/
    for (j = 0; j < plan->mv2.M_total; j++)
        for (t = 0; t < plan->mv2.d; t++)
            plan->mv2.x[plan->mv2.d * j + t] = -plan->y[plan->mv2.d * j + t]; /* note the factor -1 for conjugated transform instead of standard*/

    /** precompute psi, the entries of the matrix B */
    if (plan->mv2.nfft_flags & PRE_LIN_PSI)
        nfft_precompute_lin_psi(&(plan->mv2));

    if (plan->mv2.nfft_flags & PRE_PSI)
        nfft_precompute_psi(&(plan->mv2));

    if (plan->mv2.nfft_flags & PRE_FULL_PSI)
        nfft_precompute_full_psi(&(plan->mv2));

    /** precompute Fourier coefficients of regularised kernel*/


    for (j = 0; j < plan->n_total; j++) {

        k = j;
        plan->b[j] = 0.0;
        for (t = 0; t < 3; t++) {
            plan->b[j] += ((double) (k % (plan->n)) / (plan->n) - 0.5) * ((double) (k % (plan->n)) / (plan->n) - 0.5);
            k = k / (plan->n);
        }

        plan->b[j] = regkern3(sqrt(plan->b[j]), plan->p, plan->eps_I, plan->eps_B) / plan->n_total;

    }


    nfft_fftshift_complex(plan->b, plan->mv1.d, plan->mv1.N);
    fftw_execute(plan->fft_plan);
    nfft_fftshift_complex(plan->b, plan->mv1.d, plan->mv1.N);

}

/** fast NFFT-based summation */
void fastsum_trafo(fastsum_plan *plan) {
    int j, k,t;
    double ymin[3], ymax[3];

    nfft_adjoint(&(plan->mv1));

    for (k = 0; k < plan->mv2.N_total; k++)
        plan->mv2.f_hat[k] = plan->b[k] * plan->mv1.f_hat[k];

    nfft_trafo(&(plan->mv2));

    for (j = 0; j < plan->N_target; j++) {

        for (t=0; t<3; t++)
      {
        ymin[t] = plan->y[3*j+t] - plan->eps_I;
        ymax[t] = plan->y[3*j+t] + plan->eps_I;
      }
        plan->f[j] = plan->mv2.f[j]+SearchTree(0, plan->x, plan->alpha, ymin, ymax, plan->N_source, plan->Ad, plan->Add, plan->p);

    }


}
