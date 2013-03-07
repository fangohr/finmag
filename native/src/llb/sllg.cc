#include "finmag_includes.h"

#include "util/np_array.h"

#include "mt19937.h"

#include "llb.h"


namespace finmag { namespace llb {

    class StochasticSLLGIntegrator {

        double theta;
        double theta1;
        double theta2;

    	private:
        	int length;
        	np_array<double> M,M_pred,Ms_arr,T_arr,V_arr,alpha_arr;
        	double dt,gamma,Q;
        	double *dm1, *dm2, *dm3, *eta;
        	bp::object rhs_func;
        	unsigned int seed;
        	RandomMT19937 mt_random;
        	bool check_magnetisation_length;

        	void (StochasticSLLGIntegrator::*run_step_fun)(const np_array<double> &H);

        	void calc_llg_adt_bdw(double *m,double *h,double *dm);
        	void run_step_rk2(const np_array<double> &H);
        	void run_step_rk3(const np_array<double> &H);
        	void check_normalise();

    	public:
        	StochasticSLLGIntegrator(
        			const np_array<double> &M,
        			const np_array<double> &M_pred,
        			const np_array<double> &Ms,
        			const np_array<double> &T,
        			const np_array<double> &V,
					const np_array<double> &alpha,
					const bp::object _rhs_func,
					const std::string method_name);

        	~StochasticSLLGIntegrator();

        	void set_parameters(double dt,double gamma,unsigned int seed, bool checking);
        	void run_step(const np_array<double> &H);
    };


    StochasticSLLGIntegrator::~StochasticSLLGIntegrator(){

    	if (dm1!=0){
    		delete[] dm1;
    	}

    	if (dm2!=0){
    		delete[] dm2;
    	}

    	if (dm3!=0){
    		delete[] dm3;
    	}

    	if (eta!=0){
    	    delete[] eta;
    	}

    }



    StochasticSLLGIntegrator::StochasticSLLGIntegrator(
    							const np_array<double> &M,
    							const np_array<double> &M_pred,
    							const np_array<double> &Ms,
    							const np_array<double> &T,
    							const np_array<double> &V,
    							const np_array<double> &alpha,
    					        bp::object _rhs_func,
    							std::string method_name):
    							M(M),
    							M_pred(M_pred),
    							Ms_arr(Ms),
    							T_arr(T),
    							V_arr(V),
    							alpha_arr(alpha),
    							rhs_func(_rhs_func){

        							assert(M.size()==3*T.size());
        							assert(M_pred.size()==M.size());

        							length=M.size();

        							dm1= new double[length];
        							dm2= new double[length];
        							eta= new double[length];

        							if (_rhs_func.is_none())
        								throw std::invalid_argument("StochasticSLLGIntegrator: _rhs_func is None");

        							if (method_name=="RK2a"){
        								run_step_fun=&StochasticSLLGIntegrator::run_step_rk2;
        								theta=1.0;
        						        theta1=0.5;
        						        theta2=0.5;
        							}else if(method_name=="RK2b"){
        								run_step_fun=&StochasticSLLGIntegrator::run_step_rk2;
        								theta=2.0/3.0;
        								theta1=0.25;
        								theta2=0.75;
        							}else if(method_name=="RK2c"){
        								run_step_fun=&StochasticSLLGIntegrator::run_step_rk2;
        								theta=0.5;
        								theta1=0;
        								theta2=1.0;
        							}else if(method_name=="RK3"){
        								run_step_fun=&StochasticSLLGIntegrator::run_step_rk3;
        								dm3= new double[length];
        							}else{
        								throw std::invalid_argument("StochasticSLLGIntegrator:Only RK2a, RK2b, RK2c and RK3 are implemented!");
        							}


        }


    void StochasticSLLGIntegrator::run_step(const np_array<double> &H) {

    	(this->*run_step_fun)(H);

    }

    void StochasticSLLGIntegrator::check_normalise(){
    	double *m = M.data();
    	int len=length/3;

    	int i,j,k;

    	double max_m=0;
    	double mm;

    	for (i = 0; i < len; i++) {
    		j = i + len;
    		k = j + len;
    		mm = sqrt(m[i] * m[i] + m[j] * m[j] + m[k] * m[k]);
    		if (mm>max_m){
    			max_m=mm;
    		}

    		mm=1.0/mm;
    		m[i] *= mm;
    		m[j] *= mm;
    		m[k] *= mm;
    	}

    	if (check_magnetisation_length){

    		if (max_m>1.05 || max_m<0.95){
    			std::ostringstream ostr;
    			ostr << "maxm=" << max_m <<", so dt="<< dt << " is probably too large!";
    			throw std::invalid_argument(ostr.str());
    		}
    	}


    }

    void StochasticSLLGIntegrator::run_step_rk2(const np_array<double> &H) {

    		double *h = H.data();
    		double *m = M.data();
    		double *m_pred=M_pred.data();

    		bp::call<void>(rhs_func.ptr(),M);

    		mt_random.gaussian_random_vec(eta,length,sqrt(dt));
    		calc_llg_adt_bdw(m,h,dm1);

    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] + theta*dm1[i];
    		}

    		bp::call<void>(rhs_func.ptr(),M_pred);

    		calc_llg_adt_bdw(m_pred,h,dm2);

    		for (int i = 0; i < length; i++){
    			m[i] += theta1*dm1[i] + theta2*dm2[i];
    		}

    		check_normalise();

    }

    void StochasticSLLGIntegrator::run_step_rk3(const np_array<double> &H) {
    		double *h = H.data();
    		double *m = M.data();
    		double *m_pred=M_pred.data();
    		double two_three=2.0/3.0;

    		mt_random.gaussian_random_vec(eta,length,sqrt(dt));

    		bp::call<void>(rhs_func.ptr(),M);
    		calc_llg_adt_bdw(m,h,dm1);
    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] + two_three*dm1[i];
    		}

    		bp::call<void>(rhs_func.ptr(),M_pred);
    		calc_llg_adt_bdw(m_pred,h,dm2);
    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] - dm1[i]+ dm2[i];
    		}

    		bp::call<void>(rhs_func.ptr(),M_pred);
    		calc_llg_adt_bdw(m_pred,h,dm3);
    		for (int i = 0; i < length; i++){
    			m[i] += 0.75*dm2[i] + 0.25*dm3[i];
    		}

    		check_normalise();

    }

    void StochasticSLLGIntegrator::set_parameters(double dt,double gamma,unsigned int seed,bool checking){
    	double k_B = 1.3806505e-23;
    	double mu_0 = 4 * M_PI * 1e-7;
    	this->dt=dt;
    	this->gamma=gamma;
    	this->Q = k_B / (gamma * mu_0);
    	this->seed=seed;
    	this->check_magnetisation_length=checking;
    	mt_random.seed(seed);
    }


    void StochasticSLLGIntegrator::calc_llg_adt_bdw(double *m,double *h,double *dm){

        int i,j,k;
    	double *T = T_arr.data();
        double *V = V_arr.data();
        double *Ms = Ms_arr.data();
        double *alpha=alpha_arr.data();
        double alpha_inv,q,coeff;
        double mth0,mth1,mth2;
        int len=length/3;

    	for (i = 0; i < len; i++) {
    		j = i + len;
    		k = j + len;

    		alpha_inv= 1.0/ (1.0 + alpha[i] * alpha[i]);
    		coeff = -gamma*alpha_inv ;
    		q = sqrt(2 * Q * alpha[i] *alpha_inv * T[i] / (Ms[i]* V[i]));

    		mth0 = coeff * (m[j] * h[k] - m[k] * h[j]) * dt;
    		mth1 = coeff * (m[k] * h[i] - m[i] * h[k]) * dt;
    		mth2 = coeff * (m[i] * h[j] - m[j] * h[i]) * dt;

    		mth0 += coeff * (m[j] * eta[k] - m[k] * eta[j]) * q;
    		mth1 += coeff * (m[k] * eta[i] - m[i] * eta[k]) * q;
    		mth2 += coeff * (m[i] * eta[j] - m[j] * eta[i]) * q;

    		dm[i] = mth0 + alpha[i] * (m[j] * mth2 - m[k] * mth1);
    		dm[j] = mth1 + alpha[i] * (m[k] * mth0 - m[i] * mth2);
    		dm[k] = mth2 + alpha[i] * (m[i] * mth1 - m[j] * mth0);

    	}

    }



    void register_sllg() {
    	using namespace bp;

    	class_<StochasticSLLGIntegrator>("StochasticSLLGIntegrator", init<
    			 	np_array<double>,
    			 	np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    bp::object,
    			    std::string>())
    	        	.def("run_step", &StochasticSLLGIntegrator::run_step)
    	        	.def("set_parameters", &StochasticSLLGIntegrator::set_parameters);
    }


}}

