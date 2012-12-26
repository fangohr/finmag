#ifdef __cplusplus
extern "C" {
#endif


	#include <time.h>
	#include <stdlib.h>

	#define MT19937_N		624
	#define MT19937_M		397
	#define MT19937_INIT_MULT	0x6c078965U

	#define MT19937_MASK_B 0x9d2c5680U
	#define MT19937_MASK_C 0xefc60000U
	#define MT19937_MATRIX_A 0x9908b0dfU

	#define MT19973_SHIFT_U 11
	#define MT19973_SHIFT_S 7
	#define MT19973_SHIFT_T 15
	#define MT19973_SHIFT_L 18

	#define	MT19973_RAND_MAX 4294967295u

	static unsigned int MT[MT19937_N];
	static unsigned int mt19937_matrix[2] = { 0, MT19937_MATRIX_A };
	static int finmag_random_index = 0;

	inline unsigned int int_rand(void) {

		unsigned int x;

		x = (MT[finmag_random_index] & 0x1U) + (MT[(finmag_random_index + 1) % MT19937_N] & 0xFFFFFFFEU);
		MT[finmag_random_index] = (MT[(finmag_random_index + MT19937_M) % MT19937_N] ^ (x >> 1))
				^ mt19937_matrix[x & 1];

		x = MT[finmag_random_index];
		x ^= (x >> MT19973_SHIFT_U);
		x ^= (x << MT19973_SHIFT_S) & MT19937_MASK_B;
		x ^= (x << MT19973_SHIFT_T) & MT19937_MASK_C;
		x ^= (x >> MT19973_SHIFT_L);

		finmag_random_index = (finmag_random_index + 1) % MT19937_N;

		return x;
	}

	//temporary random generator
	void initial_random(void) {
		int i;
		unsigned int seed = (unsigned int) time(NULL);

		MT[0] = seed & 0xFFFFFFFFU;
		for (i = 1; i < MT19937_N; i++) {
			MT[i] = (MT[i - 1]^ (MT[i - 1] >> 30)) + i;
			MT[i] *= MT19937_INIT_MULT;
			MT[i] &= 0xFFFFFFFFU;
		}
	}

	inline double mt19937_random(void) {
		return ((double) int_rand()) / (double) MT19973_RAND_MAX;
	}

	inline double gauss_random(void) {
		static int flag = 1;
		static double rnd;
		if (flag) {
			double x, y, r;

			do {
				x = 2 * mt19937_random() - 1;
				y = 2 * mt19937_random() - 1;
				r = x * x + y * y;
			} while (r >= 1.0);

			r = sqrt(-2.0 * log(r) / r);

			rnd = y * r;
			flag = 0;

			return x * r;
		} else {
			flag = 1;
			return rnd;
		}

	}

	static const double a[] =
	{
	        -3.969683028665376e+01,
	         2.209460984245205e+02,
	        -2.759285104469687e+02,
	         1.383577518672690e+02,
	        -3.066479806614716e+01,
	         2.506628277459239e+00
	};

	static const double b[] =
	{
	        -5.447609879822406e+01,
	         1.615858368580409e+02,
	        -1.556989798598866e+02,
	         6.680131188771972e+01,
	        -1.328068155288572e+01
	};

	static const double c[] =
	{
	        -7.784894002430293e-03,
	        -3.223964580411365e-01,
	        -2.400758277161838e+00,
	        -2.549732539343734e+00,
	         4.374664141464968e+00,
	         2.938163982698783e+00
	};

	static const double d[] =
	{
	        7.784695709041462e-03,
	        3.224671290700398e-01,
	        2.445134137142996e+00,
	        3.754408661907416e+00
	};

    #define P_LOW 0.02425
	#define P_HIGH 0.97575

	//norm random numbers, see http://home.online.no/~pjacklam/notes/invnorm/
	double ltqnorm(void) {
        double q, r;
        double p=mt19937_random();

        if (p <= 0) {
              return -MT19973_RAND_MAX;
        }
        else if (p >= 1) {
             return MT19973_RAND_MAX;
        }
        else if (p < P_LOW){
                q = sqrt(-2*log(p));
                return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        }
        else if (p > P_HIGH){
                q  = sqrt(-2*log(1-p));
                return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                        ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
        }
        else{
                q = p - 0.5;
                r = q*q;
                return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
                        (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
        }
    }


	void gauss_random_vec(double *x, int n, double dev) {
		int i;
		for (i = 0; i < n; i++) {
			//x[i] = dev * gauss_random();
			x[i] = dev * ltqnorm();
		}

	}






#ifdef __cplusplus
}
#endif
