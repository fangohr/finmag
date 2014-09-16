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


	inline double compute_dm(double *a, double *b, int n){
		double sum=0;
		double t=0;

		for(int i=0;i<n;i++){
			t = b[i]-a[i];

			if (t > M_PI){
				t = 2*M_PI - t;
			}else if(t < -M_PI){
				t += 2*M_PI;
			}

			sum += t*t;

		}

		t = sqrt(sum)/n;
		return t;

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



	void compute_springs(
			const np_array<double> &ys,
			const np_array<double> &springs,
			double const spring_coeff) {

			ys.check_ndim(2, "check_dimensions: ys");
            int const image_num = ys.dim()[0];
            int const nodes = ys.dim()[1];

            double *spring = springs.data();

            double dm1, dm2;

            for(int i=1; i<image_num-1; i++){
            	double *ya = ys(i-1);
            	double *yb = ys(i);
            	double *yc = ys(i+1);

            	dm1 = compute_dm(ya, yb, nodes);
            	dm2 = compute_dm(yb, yc, nodes);

            	spring[i-1] = spring_coeff*(dm2-dm1);

            }
	}

      void compute_dmdt(double *m, double *h, double *dm_dt, int nodes){

        		int n = nodes/3;

        		for(int i=0;i<n;i++){
        			int j=i+n;
        			int k=j+n;

        	        double mm = m[i]*m[i] + m[j]*m[j] + m[k]*m[k];
        	        double mh = m[i]*h[i] + m[j]*h[j] + m[k]*h[k];
        	        //mm.h-mh.m=-mx(mxh)
        	        dm_dt[i] = mm*h[i] - mh*m[i];
        	        dm_dt[j] = mm*h[j] - mh*m[j];
        	        dm_dt[k] = mm*h[k] - mh*m[k];

        	        double c = 6*sqrt(dm_dt[i]*dm_dt[i]+dm_dt[j]*dm_dt[j]+dm_dt[k]*dm_dt[k]);

        	        dm_dt[i] += c*(1-mm)*m[i];
        	        dm_dt[j] += c*(1-mm)*m[j];
        	        dm_dt[k] += c*(1-mm)*m[k];
        		}


       }

      void compute_dm_dt(const np_array<double> &ys,
        			const np_array<double> &heff,
        			const np_array<double> &dm_dt) {

        			ys.check_ndim(2, "check_dimensions: ys");
        			heff.check_ndim(2, "check_dimensions: heff");
        			dm_dt.check_ndim(2, "check_dimensions: dm_dt");

                    int const image_num = ys.dim()[0];
                    int const nodes = ys.dim()[1];

                    for(int i=1; i<image_num-1; i++){
                    	double *y = ys(i);
                    	double *h = heff(i);
                    	double *dmdt = dm_dt(i);

                    	compute_dmdt(y, h, dmdt, nodes);

                    }

                   return;

     }

    void register_neb() {

    	using namespace bp;

    	def("compute_tangents", &compute_tangents, (
    	            arg("ys"),
    	            arg("energies"),
    	            arg("tangents")
    	        ));

    	def("compute_springs", &compute_springs, (
    	            arg("ys"),
    	            arg("springs"),
    	            arg("spring_coeff")
    	        ));

    	def("compute_dm_dt", &compute_dm_dt, (
    	            arg("ys"),
    	            arg("heff"),
    	            arg("dm_dt")
    	        ));
    }
}}
