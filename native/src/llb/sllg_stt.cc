#include "finmag_includes.h"

#include "util/np_array.h"

#include "mt19937.h"

#include "llb.h"

#include "util/python_threading.h"


namespace finmag { namespace llb {

	double const const_e = 1.602176565e-19; // elementary charge in As
	double const mu_B = 9.27400968e-24; //Bohr magneton

    class StochasticLLGIntegratorSTT {

        double theta;
        double theta1;
        double theta2;

    	private:
        	int length;
        	np_array<double> M,M_pred,Ms_arr,T_arr,V_arr,alpha_arr;
        	double P, beta;
        	double dt,gamma,Q, u0;
        	double *dm1, *dm2, *eta;
        	bp::object rhs_func;
        	unsigned int seed;
        	RandomMT19937 mt_random;
        	bool check_magnetisation_length;

        	void calc_llg_adt_bdw(double *m,double *h,double *gh, double *dm);
        	void check_normalise();

    	public:
        	StochasticLLGIntegratorSTT(
        			const np_array<double> &M,
        			const np_array<double> &M_pred,
        			const np_array<double> &Ms,
        			const np_array<double> &T,
        			const np_array<double> &V,
					const np_array<double> &alpha,
					const double P,
					const double beta,
					const bp::object _rhs_func,
					const std::string method_name);

        	~StochasticLLGIntegratorSTT();

        	void set_parameters(double dt,double gamma,unsigned int seed, bool checking);
        	void run_step(const np_array<double> &H, const np_array<double> &grad_H);
    };


    StochasticLLGIntegratorSTT::~StochasticLLGIntegratorSTT(){

    	if (dm1!=0){
    		delete[] dm1;
    	}

    	if (dm2!=0){
    		delete[] dm2;
    	}

    	if (eta!=0){
    	    delete[] eta;
    	}

    }


    StochasticLLGIntegratorSTT::StochasticLLGIntegratorSTT(
    							const np_array<double> &M,
    							const np_array<double> &M_pred,
    							const np_array<double> &Ms,
    							const np_array<double> &T,
    							const np_array<double> &V,
    							const np_array<double> &alpha,
    							const double P,
    							const double beta,
    					        bp::object _rhs_func,
    							std::string method_name):
    							M(M),
    							M_pred(M_pred),
    							Ms_arr(Ms),
    							T_arr(T),
    							V_arr(V),
    							alpha_arr(alpha),
    							P(P),beta(beta),
    							rhs_func(_rhs_func){

        							assert(M.size()==3*T.size());
        							assert(M_pred.size()==M.size());

        							length=M.size();

        							dm1= new double[length];
        							dm2= new double[length];
        							eta= new double[length];

        							if (_rhs_func.is_none())
        								throw std::invalid_argument("StochasticLLGIntegratorSTT: _rhs_func is None");

        							if (method_name=="RK2a"){
        								theta=1.0;
        						        theta1=0.5;
        						        theta2=0.5;
        							}else if(method_name=="RK2b"){
        								theta=2.0/3.0;
        								theta1=0.25;
        								theta2=0.75;
        							}else if(method_name=="RK2c"){
        								theta=0.5;
        								theta1=0;
        								theta2=1.0;
        							}else{
        								throw std::invalid_argument("StochasticLLGIntegratorSTT:Only RK2a, RK2b and RK2c are implemented!");
        							}

        			this->u0 = P*mu_B/const_e; // P g mu_B/(2 e Ms) and g=2 for electrons

        			//printf("P=%g  beta=%g  u0=%g\n", P, beta, u0);


        }

    void StochasticLLGIntegratorSTT::check_normalise(){
    	double *m = M.data();
    	int len=length/3;

    	int i,j,k;

    	double mm;

    	for (i = 0; i < len; i++) {
    		j = i + len;
    		k = j + len;
    		mm = sqrt(m[i] * m[i] + m[j] * m[j] + m[k] * m[k]);

    		mm=1.0/mm;
    		m[i] *= mm;
    		m[j] *= mm;
    		m[k] *= mm;
    	}

    }

    void StochasticLLGIntegratorSTT::run_step(const np_array<double> &H, const np_array<double> &grad_H) {

    		double *h = H.data();
    		double *hg = grad_H.data();
    		double *m = M.data();
    		double *m_pred=M_pred.data();

    		bp::call<void>(rhs_func.ptr(),M);

    		mt_random.gaussian_random_vec(eta,length,sqrt(dt));
    		calc_llg_adt_bdw(m,h,hg,dm1);

    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] + theta*dm1[i];
    		}

    		bp::call<void>(rhs_func.ptr(),M_pred);

    		calc_llg_adt_bdw(m_pred,h,hg,dm2);

    		for (int i = 0; i < length; i++){
    			m[i] += theta1*dm1[i] + theta2*dm2[i];
    		}

    		check_normalise();

    }

    void StochasticLLGIntegratorSTT::set_parameters(double dt,double gamma,unsigned int seed,bool checking){
    	double k_B = 1.3806505e-23;
    	double mu_0 = 4 * M_PI * 1e-7;

    	this->dt=dt;
    	this->gamma=gamma;
    	this->Q = k_B / (gamma * mu_0);
    	this->seed=seed;
    	this->check_magnetisation_length=checking;

    	mt_random.initial_random(seed);
    }


    void StochasticLLGIntegratorSTT::calc_llg_adt_bdw(double *m, double *h, double *hg, double *dm){

    	double *T = T_arr.data();
        double *V = V_arr.data();
        double *Ms = Ms_arr.data();
        double *alpha = alpha_arr.data();
        int len = length/3;

        finmag::util::scoped_gil_release release_gil;

		#pragma omp parallel for schedule(guided)
    	for (int i = 0; i < len; i++) {
    		int j = i + len;
    		int k = j + len;

    		double alpha_inv= 1.0/ (1.0 + alpha[i] * alpha[i]);
    		double coeff = -gamma * alpha_inv ;
    		double q = sqrt(2 * Q * alpha[i] *alpha_inv * T[i] / (Ms[i]* V[i]));

    		double mth0 = coeff * (m[j] * h[k] - m[k] * h[j]) * dt;
    		double mth1 = coeff * (m[k] * h[i] - m[i] * h[k]) * dt;
    		double mth2 = coeff * (m[i] * h[j] - m[j] * h[i]) * dt;

    		mth0 += coeff * (m[j] * eta[k] - m[k] * eta[j]) * q;
    		mth1 += coeff * (m[k] * eta[i] - m[i] * eta[k]) * q;
    		mth2 += coeff * (m[i] * eta[j] - m[j] * eta[i]) * q;

    		dm[i] = mth0 + alpha[i] * (m[j] * mth2 - m[k] * mth1);
    		dm[j] = mth1 + alpha[i] * (m[k] * mth0 - m[i] * mth2);
    		dm[k] = mth2 + alpha[i] * (m[i] * mth1 - m[j] * mth0);

    		// the above is the normal LLG equation

    		double coeff_stt = u0 * alpha_inv * dt;

    		if (Ms[i]==0){
    			coeff_stt = 0;
    		}else{
    			coeff_stt/=Ms[i];
    		}

    		double mht = m[i]*hg[i] + m[j]*hg[j] + m[k]*hg[k];

    		double hpi = hg[i] - mht*m[i];
    		double hpj = hg[j] - mht*m[j];
    		double hpk = hg[k] - mht*m[k];

    		mth0 = (m[j] * hpk - m[k] * hpj);
    		mth1 = (m[k] * hpi - m[i] * hpk);
    		mth2 = (m[i] * hpj - m[j] * hpi);

    		dm[i] += coeff_stt*((1 + alpha[i]*beta)*hpi - (beta - alpha[i])*mth0);
    		dm[j] += coeff_stt*((1 + alpha[i]*beta)*hpj - (beta - alpha[i])*mth1);
    		dm[k] += coeff_stt*((1 + alpha[i]*beta)*hpk - (beta - alpha[i])*mth2);

    	}

    }


    void register_sllg_stt() {

    	using namespace bp;

    	class_<StochasticLLGIntegratorSTT>("StochasticLLGIntegratorSTT", init<
    			 	np_array<double>,
    			 	np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    double,
    			    double,
    			    bp::object,
    			    std::string>())
    	        	.def("run_step", &StochasticLLGIntegratorSTT::run_step)
    	        	.def("set_parameters", &StochasticLLGIntegratorSTT::set_parameters);
    }


}}

