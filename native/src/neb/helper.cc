#include "finmag_includes.h"

#include "util/np_array.h"

#include "helper.h"


namespace finmag { namespace neb {

	//normalise the given array
	inline void normalise(double *res, int n){

		double length=0;

		for(int i=0;i<n;i++){

			if (res[i] > M_PI){
				res[i] = 2*M_PI - res[i];
			}else if(res[i] < -M_PI){
				res[i] += 2*M_PI;
			}

			length += res[i]*res[i];
		}

		if (length>0){
			length = 1.0/sqrt(length);
		}

		for(int i=0;i<n;i++){
			res[i]*=length;
		}

	}

	//compute b-a and store it to res with normalised length 1
	inline void difference(double *res, double *a, double *b, int n){

		for(int i=0;i<n;i++){
			res[i] = b[i]-a[i];
		}

		normalise(res, n);
	}


	void compute_tangents(
			const np_array<double> &ys,
			const np_array<double> &energies,
			const np_array<double> &tangents) {

			ys.check_ndim(2, "check_dimensions: ys");
            int const image_num = ys.dim()[0];
            int const nodes = ys.dim()[1];

            //printf("%d   %d\n", image_num, nodes);

            tangents.check_ndim(2, "check_dimensions: tangents");

            double t1[nodes];
            double t2[nodes];

            double *energy = energies.data();

            for(int i=1; i<image_num-1; i++){
            	double *ya = ys(i-1);
            	double *yb = ys(i);
            	double *yc = ys(i+1);
            	double *t = tangents(i-1);

            	double e1 = energy[i-1]-energy[i];
            	double e2 = energy[i]-energy[i+1];

            	if (e1<0 && e2<0){
            		difference(t, yb, yc, nodes);
            	}else if(e1>0&&e2>0){
            		difference(t, ya, yb, nodes);
            	}else{
            		difference(t1, ya, yb, nodes);
            		difference(t2, yb, yc, nodes);

            		double max_e, min_e;

            		if (fabs(e1)>fabs(e2)){
            			max_e = fabs(e1);
            			min_e = fabs(e2);
            		}else{
            			max_e = fabs(e2);
            			min_e = fabs(e1);
            		}

            		if (energy[i+1]>energy[i-1]){
            			for(int i=0;i<nodes;i++){
            				t[i] = min_e*t1[i] + max_e*t2[i];
            			}
            		}else{
            			for(int i=0;i<nodes;i++){
            				t[i] = max_e*t1[i] + min_e*t2[i];
            			}
            		}

            		normalise(t, nodes);
            	}


            }

	}



    void register_neb() {

    	using namespace bp;

    	def("compute_tangents", &compute_tangents, (
    	            arg("ys"),
    	            arg("energies"),
    	            arg("tangents")
    	        ));

    }
}}
