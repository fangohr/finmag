#include "finmag_includes.h"

#include "util/np_array.h"

#include "llb.h"


namespace finmag { namespace llb {

	static const double const_MU0 = M_PI*4e-7; // T m/A
	static const double const_K_B = 1.3806488e-23; // J/K


    class Materials {

    	private:
        	double Ms0;
        	double Tc;
        	double A0;
        	double K0;
        	double mu_a;//atomistic magnetic moment


    	public:
        	Materials(double Ms,double Tc,double A,double K, double mu_a){
        		this->Ms0=Ms;
        		this->Tc=Tc;
        		this->A0=A;
        		this->K0=K;
        		this->mu_a=mu_a;
        	};
        	double m_e(double T);
        	double Ms(double T);
        	double A(double T);
        	double chi_par(double T);
        	double chi_perp(double T);
        	double inv_chi_par(double T);
        	double inv_chi_perp(double T);
        	void compute_relaxation_field(
 	           	   const np_array<double> & T_arr,
 	           	   const np_array<double> & M,
 	           	   const np_array<double> & H);

    };

    double Materials::m_e(double T){

    	double a0=1.316787568985625;
    	double a1=-0.170703768212514;
    	double a2=-0.303750348990166;
    	double a3=0.157666548217993;

    	double t=1.0-T/Tc;
    	if (t<=0){
    		return 0;
    	}else if (t>=1){
    		return 1.0;
    	}

    	double res=a0*sqrt(t)+t*(a1+t*a2+a3*t*t);

    	return res;
    }

    double Materials::Ms(double T){
    	return Ms0*m_e(T);
    }

    double Materials::A(double T){
    	double a0=0.327601417432234;
    	double a1=1.446591232606010;
    	double a2=-1.402497304201748;
    	double a3=0.628304654164059;

    	double t=1.0-T/Tc;

    	if (t<=0){
    		return 0;
    	}else if (t>=1){
    		return A0;
    	}

    	double res=a0*sqrt(t)+t*(a1+t*a2+a3*t*t);

    	return res*A0;
    }


    double Materials::chi_par(double T){

    			double eps=1e-3;
    			double eps2=1e-3;
    			double x=T/Tc;
    			double res=0;

    			if (x<1){
    				double a0=0.153298742094880;
    				double a1=-1.249707138220772;
    				double a2=2.809708026822461;
    				double a3=-2.537192200789637;
    				double a4=0.824002473372640;

    				if (x<eps){
    					x=eps;
    				}else if (x>1-eps){
    					x=1-eps;
    				}

    				double t=1-x;

    				res=a0/t+t*(a1+t*(a2+a3*t+a4*t*t));

    			}else{
    				if (x<1+eps2){
    					x=1+eps2;
    				}
    				res=0.333333333333333/(x-1.0);
    			}

    			return res*(const_MU0*mu_a/(const_K_B*Tc));

            }


    double Materials::chi_perp(double T){

    			double a0=0.002759963555555556;

    			double x=T/Tc;

    			if(x<0){
    				x=0;
    			}

    			double res=(1+a0*x)*const_MU0*Ms0/(2*K0);

    			if (x>1){

    				double res2=chi_par(T);
    				if (res2<res){
    					res=res2;
    				}

    			}

    			return res;
            }

    double Materials::inv_chi_par(double T){
    	return 1.0/chi_par(T);
    }

    double Materials::inv_chi_perp(double T){
    	return 1.0/chi_perp(T);
    }


    void Materials::compute_relaxation_field(
    	           	   const np_array<double> & T_arr, //input
    	           	   const np_array<double> & M,
    	           	   const np_array<double> & H){ //output


    	            	int length=T_arr.size();
    	            	assert(length*3==M.size());

    	            	double *T = T_arr.data();
    	            	double *m=M.data();
    	            	double *h=H.data();

    	            	int i2,i3;
    	            	for (int i1 = 0; i1 < length; i1++) {
    	            		i2=length+i1;
    	            		i3=length+i2;

    	            		double temp = T[i1];
    	            		// calculate the relaxation coefficient
    	            		double me = this->m_e(temp);
    	            		double me_sq = me * me;
    	            		double r;
    	            		double m_sq = m[i1]*m[i1] + m[i2]*m[i2] + m[i3]*m[i3];

    	            		if (temp <= Tc) {
    	            			r = 0.5 * (1. - m_sq/me_sq);
    	            		} else {
    	            			r = -1. - 0.6 * Tc / (temp - Tc) * m_sq;
    	            		}

    	            		double coeff = r * inv_chi_par(temp);

    	            		h[i1] = coeff * m[i1];
    	            		h[i2] = coeff * m[i2];
    	            		h[i3] = coeff * m[i3];

    	            	}
    	         }


    void register_llb_material() {
        using namespace bp;

        class_<Materials>("Materials", init<
        		double,double,double,double,double>())
        		.def("m_e", &Materials::m_e)
        		.def("Ms", &Materials::Ms)
        		.def("A", &Materials::A)
        		.def("chi_par", &Materials::chi_par)
        		.def("chi_perp", &Materials::chi_perp)
        		.def("inv_chi_par", &Materials::inv_chi_par)
        		.def("inv_chi_perp", &Materials::inv_chi_perp)
        		.def("compute_relaxation_field",&Materials::compute_relaxation_field);
    }


}}
