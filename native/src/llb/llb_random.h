namespace finmag {

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
	static int index = 0;

	inline unsigned int int_rand() {

		unsigned int x;

		x = (MT[index] & 0x1U) + (MT[(index + 1) % MT19937_N] & 0xFFFFFFFEU);
		MT[index] = (MT[(index + MT19937_M) % MT19937_N] ^ (x >> 1))
				^ mt19937_matrix[x & 1];

		x = MT[index];
		x ^= (x >> MT19973_SHIFT_U);
		x ^= (x << MT19973_SHIFT_S) & MT19937_MASK_B;
		x ^= (x << MT19973_SHIFT_T) & MT19937_MASK_C;
		x ^= (x >> MT19973_SHIFT_L);

		index = (index + 1) % MT19937_N;

		return x;
	}

	//temporary random generator
	void initial_random() {
		unsigned int seed = (unsigned int) time(NULL);
		seed=100;
		MT[0] = seed & 0xFFFFFFFFU;
		for (int i = 1; i < MT19937_N; i++) {
			MT[i] = (MT[i - 1]^ (MT[i - 1] >> 30)) + i;
			MT[i] *= MT19937_INIT_MULT;
			MT[i] &= 0xFFFFFFFFU;
		}
	}

	inline double random() {
		return ((double) int_rand()) / (double) MT19973_RAND_MAX;
	}

	inline double gauss_random() {
		static int flag = 1;
		static double rnd;
		if (flag) {
			double x, y, r;

			do {
				x = 2 * random() - 1;
				y = 2 * random() - 1;
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

	void gauss_random_vec(double *x, int n, double dev) {
		for (int i = 0; i < n; i++) {
			x[i] = dev * gauss_random();
		}

	}
}
