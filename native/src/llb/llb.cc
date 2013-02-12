#include "finmag_includes.h"

#include "util/np_array.h"

#include "mt19937.h"

#include "llb.h"


namespace finmag { namespace llb {

	static const double constant_MU0 = M_PI*4e-7; // T m/A
	static const double constant_K_B = 1.3806488e-23; // J/K

	inline double alpha_perp(double T, double T_C) { return T <= T_C ? 1. - (1./3.)*(T/T_C) : (2./3.) * (T/T_C); }
	inline double alpha_par(double T, double T_C) { return (2./3.) * (T/T_C); }

	void calc_llb_dmdt(
			const np_array<double> &M,
			const np_array<double> &H,
			const np_array<double> &dmdt,
			const np_array<double> &T_arr,
			double gamma_LL,
			double lambda,
			double Tc,
			bool do_precession) {


		int length=T_arr.size();
		double *m=M.data();
		double *h=H.data();
		double *dm_dt=dmdt.data();
		double *T = T_arr.data();

		double precession_coeff = -gamma_LL;
		double damping_coeff = -gamma_LL*lambda;

		int i,j,k;
		// calculate dm
    	//#pragma omp parallel for schedule(guided)
		for (i = 0; i < length; i++) {
			j=length+i;
			k=length+j;
			// add precession: m x H, multiplied by -gamma
			if (do_precession) {
				dm_dt[i] = precession_coeff*(m[j]*h[k]-m[k]*h[j]);
				dm_dt[j] = precession_coeff*(m[k]*h[i]-m[i]*h[k]);
				dm_dt[k] = precession_coeff*(m[i]*h[j]-m[j]*h[i]);
			} else {
				dm_dt[i] = 0.;
				dm_dt[j] = 0.;
				dm_dt[k] = 0.;
			}

			// add transverse damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma lambda alpha_perp/m^2
			// add longitudinal damping: (m.H)m, muliplied by gamma lambdaa alpha_par/m^2
			// total damping is -gamma lambda [ (alpha_perp - alpha_par)/m^2 (m.H) m - alpha_perp H ]
			double mh = m[i] * h[i] + m[j] * h[j] + m[k] * h[k];
			double mm = m[i] * m[i] + m[j] * m[j] + m[k] * m[k];

			double a1 = alpha_perp(T[i], Tc);
			double a2 = alpha_par(T[i], Tc);
			double damp1 = (a1 - a2) * damping_coeff / mm * mh;
			double damp2 = - a1 * damping_coeff;
			dm_dt[i] += m[i] * damp1 + h[i] * damp2;
			dm_dt[j] += m[j] * damp1 + h[j] * damp2;
			dm_dt[k] += m[k] * damp1 + h[k] * damp2;
		}
	}



    class StochasticLLBIntegrator {

        double theta;
        double theta1;
        double theta2;

    	private:
        	int length;
        	np_array<double> M,M_pred,Ms_arr,T_arr,V_arr;
        	np_array<int>pins_arr;
        	double dt,Tc,lambda,Ms, gamma_LL;
        	double *dm1, *dm2, *dm3, *eta_perp, *eta_par;
        	bp::object rhs_func;
        	unsigned int seed;
        	RandomMT19937 mt_random;
        	bool do_precession;
        	bool using_type_II;

        	void (StochasticLLBIntegrator::*run_step_fun)(const np_array<double> &H);

        	void calc_llb_adt_bdw(double *m,double *h,double *dm);
        	void run_step_rk2(const np_array<double> &H);
        	void run_step_rk3(const np_array<double> &H);

    	public:
        	StochasticLLBIntegrator(
        			const np_array<double> &M,
        			const np_array<double> &M_pred,
        			const np_array<double> &Ms,
        			const np_array<double> &T,
        			const np_array<double> &V,
					const np_array<int> &pins,
					const bp::object _rhs_func,
					const std::string method_name);

        	~StochasticLLBIntegrator();

        	void set_parameters(double dt,double gamma,double lambda, double Tc,
        	    			unsigned int seed,bool do_precession, bool using_type_II);
        	void run_step(const np_array<double> &H);
    };


    StochasticLLBIntegrator::~StochasticLLBIntegrator(){

    	if (dm1!=0){
    		delete[] dm1;
    	}

    	if (dm2!=0){
    		delete[] dm2;
    	}

    	if (dm3!=0){
    		delete[] dm3;
    	}

    	if (eta_par!=0){
    	    delete[] eta_par;
    	}

    	if (eta_perp!=0){
    	    delete[] eta_perp;
    	}

    }



    StochasticLLBIntegrator::StochasticLLBIntegrator(
    							const np_array<double> &M,
    							const np_array<double> &M_pred,
    							const np_array<double> &Ms,
    							const np_array<double> &T,
    							const np_array<double> &V,
    							const np_array<int> &pins,
    					        bp::object _rhs_func,
    							std::string method_name):
    							M(M),
    							M_pred(M_pred),
    							Ms_arr(Ms),
    							T_arr(T),
    							V_arr(V),
    							pins_arr(pins),
    							rhs_func(_rhs_func){

        							assert(M.size()==3*T.size());
        							assert(M_pred.size()==M.size());

        							length=M.size();

        							dm1= new double[length];
        							dm2= new double[length];
        							eta_par= new double[length];
        							eta_perp= new double[length];

        							if (_rhs_func.is_none())
        								throw std::invalid_argument("StochasticLLBIntegrator: _rhs_func is None");

        							if (method_name=="RK2a"){
        								run_step_fun=&StochasticLLBIntegrator::run_step_rk2;
        								theta=1.0;
        						        theta1=0.5;
        						        theta2=0.5;
        							}else if(method_name=="RK2b"){
        								run_step_fun=&StochasticLLBIntegrator::run_step_rk2;
        								theta=2.0/3.0;
        								theta1=0.25;
        								theta2=0.75;
        							}else if(method_name=="RK2c"){
        								run_step_fun=&StochasticLLBIntegrator::run_step_rk2;
        								theta=0.5;
        								theta1=0;
        								theta2=1.0;
        							}else if(method_name=="RK3"){
        								run_step_fun=&StochasticLLBIntegrator::run_step_rk3;
        								dm3= new double[length];
        							}else{
        								throw std::invalid_argument("StochasticLLBIntegrator:Only RK2a, RK2b, RK2c and RK3 are implemented!");
        							}


        }


    void StochasticLLBIntegrator::run_step(const np_array<double> &H) {

    	(this->*run_step_fun)(H);

    }

    void StochasticLLBIntegrator::run_step_rk2(const np_array<double> &H) {

    		double *h = H.data();
    		double *m = M.data();
    		double *m_pred=M_pred.data();

    		bp::call<void>(rhs_func.ptr(),M);

    		mt_random.gaussian_random_vec(eta_par,length,sqrt(dt));
    		mt_random.gaussian_random_vec(eta_perp,length,sqrt(dt));

    		calc_llb_adt_bdw(m,h,dm1);

    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] + theta*dm1[i];
    		}

    		bp::call<void>(rhs_func.ptr(),M_pred);

    		calc_llb_adt_bdw(m_pred,h,dm2);

    		for (int i = 0; i < length; i++){
    			m[i] += theta1*dm1[i] + theta2*dm2[i];
    		}

    }

    void StochasticLLBIntegrator::run_step_rk3(const np_array<double> &H) {
    		double *h = H.data();
    		double *m = M.data();
    		double *m_pred=M_pred.data();
    		double two_three=2.0/3.0;

    		mt_random.gaussian_random_vec(eta_par,length,sqrt(dt));
    		mt_random.gaussian_random_vec(eta_perp,length,sqrt(dt));

    		bp::call<void>(rhs_func.ptr(),M);
    		calc_llb_adt_bdw(m,h,dm1);
    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] + two_three*dm1[i];
    		}

    		bp::call<void>(rhs_func.ptr(),M_pred);
    		calc_llb_adt_bdw(m_pred,h,dm2);
    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] - dm1[i]+ dm2[i];
    		}

    		bp::call<void>(rhs_func.ptr(),M_pred);
    		calc_llb_adt_bdw(m_pred,h,dm3);
    		for (int i = 0; i < length; i++){
    			m[i] += 0.75*dm2[i] + 0.25*dm3[i];
    		}

    }

    void StochasticLLBIntegrator::set_parameters(double dt,double gamma,double lambda, double Tc,
    			unsigned int seed,bool do_precession, bool using_type_II){
    	this->dt=dt;
    	this->gamma_LL=gamma;
    	this->lambda=lambda;
    	this->Tc=Tc;
    	this->seed=seed;
    	this->do_precession=do_precession;
    	this->using_type_II=using_type_II;
    	mt_random.seed(seed);
    }

    void StochasticLLBIntegrator::calc_llb_adt_bdw(double *m, double *h, double *dm) {

            double *T = T_arr.data();
            double *V = V_arr.data();
            double *Ms = Ms_arr.data();
            int len=T_arr.size();

            double precession_coeff = -gamma_LL;
            double damping_coeff = gamma_LL*lambda;

            // calculate dm
            int i,j,k;
            for (i = 0; i < len; i++) {
            	i=i;
            	j=len+i;
            	k=len+j;

                // add precession: m x H, multiplied by -gamma
                if (do_precession) {
                    dm[i] = precession_coeff*dt*(m[j]*h[k]-m[k]*h[j]);
                    dm[j] = precession_coeff*dt*(m[k]*h[i]-m[i]*h[k]);
                    dm[k] = precession_coeff*dt*(m[i]*h[j]-m[j]*h[i]);
                } else {
                    dm[i] = 0.;
                    dm[j] = 0.;
                    dm[k] = 0.;
                }


                double mh = m[i] * h[i] + m[j] * h[j] + m[k] * h[k];
                double mm = m[i] * m[i] + m[j] * m[j] + m[k] * m[k];

                //m x (m x H) == (m.H)m - (m.m)H,
                double a_perp = alpha_perp(T[i], Tc);
                double a_par = alpha_par(T[i], Tc);
                double damp1 = (a_par-a_perp) * damping_coeff / mm * mh;
                double damp2 = a_perp * damping_coeff;
                dm[i] += (m[i] * damp1 + h[i] * damp2)*dt;
                dm[j] += (m[j] * damp1 + h[j] * damp2)*dt;
                dm[k] += (m[k] * damp1 + h[k] * damp2)*dt;


                double Q_perp =  sqrt(
                        (2.*constant_K_B)*T[i]*(a_perp - a_par) /
                        (gamma_LL * Ms[i]* constant_MU0* V[i]* a_perp * a_perp * lambda)
                    );

                double Q_par =sqrt(
                        (2.*gamma_LL*constant_K_B*lambda)*T[i]*a_par /
                        (Ms[i] * V[i]*constant_MU0)
                    );

                //printf("Q_par==%e  Q_perp==%e \n",Q_par,Q_perp);
                double meta_perp = m[i] * eta_perp[i] + m[j] * eta_perp[j] + m[k] * eta_perp[k];
                //m x (m x H) == (m.H)m - (m.m)H,
                double damp3 = -a_perp * damping_coeff / mm * meta_perp;
                dm[i] += (m[i] * damp3 + eta_perp[i] * damp2)*Q_perp;
                dm[j] += (m[j] * damp3 + eta_perp[j] * damp2)*Q_perp;
                dm[k] += (m[k] * damp3 + eta_perp[k] * damp2)*Q_perp;

                if (using_type_II){
                	dm[i] += Q_par*eta_par[i];
                	dm[j] += Q_par*eta_par[j];
                	dm[k] += Q_par*eta_par[k];
                }else{
                	double meta_par = m[i] * eta_par[i] + m[j] * eta_par[j] + m[k] * eta_par[k];
                	double damp4 = a_par * damping_coeff / mm * meta_par * Q_par;
                	dm[i] += eta_par[i]* m[i] * damp4;
                	dm[j] += eta_par[j]* m[j] * damp4;
                	dm[k] += eta_par[k]* m[k] * damp4;;
                }

            }
        }




    void register_llb() {
    	using namespace bp;

    	def("calc_llb_dmdt", &calc_llb_dmdt, (
    	            arg("M"),
    	            arg("H"),
    	            arg("dmdt"),
    	            arg("T"),
    	            arg("gamma_LL"),
    	            arg("lambda"),
    	            arg("Tc"),
    	            arg("do_precession")
    	        ));

    	class_<StochasticLLBIntegrator>("StochasticLLBIntegrator", init<
    			 	np_array<double>,
    			 	np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    np_array<int>,
    			    bp::object,
    			    std::string>())
    	        	.def("run_step", &StochasticLLBIntegrator::run_step)
    	        	.def("set_parameters", &StochasticLLBIntegrator::set_parameters);
    }


}}
