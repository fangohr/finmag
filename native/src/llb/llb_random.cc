#include "finmag_includes.h"

#include "util/np_array.h"

#include "llb.h"


namespace finmag { namespace llb {

namespace {

	//temporary random generator
	void initial_random(){
		unsigned int seed = (unsigned int)time(NULL);
		srand (seed);
	}

	inline double random(){
		return (double)rand() / (double)RAND_MAX ;
	}

	inline double gauss_random(){
		static int flag=1;
		static double rnd;
		if (flag){
			double x,y,r;

			do{
				x = 2 * random() - 1;
				y = 2 * random() - 1;
				r = x * x + y * y;
			} while (r>=1.0);

			r = sqrt(-2.0 * log(r)/r);

			rnd = y * r;
			flag=0;

			return x*r;
		}
		else{
			flag=1;
			return rnd;
		}

	}

}

	void register_llb_random() {
    	using namespace bp;

    	def("initial_random",&initial_random);

    	def("random",&random);

    	def("gauss_random",&gauss_random);

    }

}}
