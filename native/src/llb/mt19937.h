//#include <boost/random.hpp>
//#include <boost/random/normal_distribution.hpp>

#include "util/np_array.h"


namespace finmag { namespace llb {

	/*
	class RandomMT19937 {

		private:
    		boost::random::mt19937 engine;
    		boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > generator;

		public:
			RandomMT19937():engine(),
							generator(engine, boost::normal_distribution<>(0.0, 1.0)) {}

			void seed(unsigned int sd) {
					engine.seed(sd);
			}

			void gaussian_random_vec(double *x, int n, double dev){
				for (int i = 0; i < n; i++) {
					x[i]=dev*generator();
				}
			}

	};
	*/


		#define MT19937_N   624


        class RandomMT19937 {

                int random_index;
                unsigned int MT[MT19937_N];

                private:
                        double ltqnorm(void);

                public:
                        RandomMT19937():random_index(0){};
                        double random(void);
                        void initial_random(unsigned int seed);
                        void gaussian_random_vec(double *x, int n, double dev);
                        void gaussian_random_np(const np_array<double> &pa);

        };

}}

