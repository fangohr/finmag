#include "finmag_includes.h"

#include "util/np_array.h"

#include "llb_random.h"

#include "llb.h"


namespace finmag { namespace llb {


namespace {

    static const double constant_MU0 = M_PI*4e-7; // T m/A
    static const double constant_K_B = 1.3806488e-23; // J/K

    inline double alpha_perp(double T, double T_C) { return T <= T_C ? 1. - (1./3.)*(T/T_C) : (2./3.) * (T/T_C); }
    inline double alpha_par(double T, double T_C) { return (2./3.) * (T/T_C); }

    void test(double *x){
    	x[0]=1000;
    }

    void test_numpy(const np_array<double> &m){
    	printf("size of m:%d\n",m.size());
    	int length=m.size();
    	assert(length%3==0);
    	double *M=m.data();
    	for (int i=0;i<length;i++){
    		printf("m[%d]=%f\n",i,M[i]);
    		M[i]*=2;
    	}
    	test(M);
    }


    void calc_llb_dmdt(
            const np_array<double> &M,
            const np_array<double> &H,
            const np_array<double> &dmdt,
            const np_array<double> &T_arr,
            double gamma_LL,
            double lambda,

            double Tc,
            bool do_precession) {

    	//assert(M.size()==H.size());
    	//assert(M.size()%3==0);

    	int length=T_arr.size();
    	double *m=M.data();
    	double *h=H.data();
    	double *dm_dt=dmdt.data();
    	double *T = T_arr.data();



        double precession_coeff = -gamma_LL;
        double damping_coeff = -gamma_LL*lambda;


        // calculate dm
        //#pragma omp parallel for schedule(guided)
        int i2,i3;
        for (int i1 = 0; i1 < length; i1++) {
        	i2=length+i1;
        	i3=length+i2;
            // add precession: m x H, multiplied by -gamma
            if (do_precession) {
                dm_dt[i1] = precession_coeff*(m[i2]*h[i3]-m[i3]*h[i2]);
                dm_dt[i2] = precession_coeff*(m[i3]*h[i1]-m[i1]*h[i3]);
                dm_dt[i3] = precession_coeff*(m[i1]*h[i2]-m[i2]*h[i1]);
            } else {
                dm_dt[i1] = 0.;
                dm_dt[i2] = 0.;
                dm_dt[i3] = 0.;
            }

            // add transverse damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma lambda alpha_perp/m^2
            // add longitudinal damping: (m.H)m, muliplied by gamma lambdaa alpha_par/m^2
            // total damping is -gamma lambda [ (alpha_perp - alpha_par)/m^2 (m.H) m - alpha_perp H ]
            double mh = m[i1] * h[i1] + m[i2] * h[i2] + m[i3] * h[i3];
            double mm = m[i1] * m[i1] + m[i2] * m[i2] + m[i3] * m[i3];

            double a1 = alpha_perp(T[i1], Tc);
            double a2 = alpha_par(T[i1], Tc);
            double damp1 = (a1 - a2) * damping_coeff / mm * mh;
            double damp2 = - a1 * damping_coeff;
            dm_dt[i1] += m[i1] * damp1 + h[i1] * damp2;
            dm_dt[i2] += m[i2] * damp1 + h[i2] * damp2;
            dm_dt[i3] += m[i3] * damp1 + h[i3] * damp2;
        }
    }




    class HeunStochasticIntegrator {
    public:

        HeunStochasticIntegrator(
            const np_array<double> &M,
            const np_array<double> &M_pred,
            const np_array<double> &T,
            const np_array<double> &V,
            double dt,
            double gamma_LL,
            double lambda,
            double Tc,
            double Ms,
            bool do_precession,
            bool use_evans2012_noise,
            bp::object _rhs_func):
            M(M),
            M_pred(M_pred),
            T_arr(T),
            V_arr(V),
            dt(dt),
            gamma_LL(gamma_LL),
            lambda(lambda),
            Tc(Tc),
            Ms(Ms),
            do_precession(do_precession),
            use_evans2012_noise(use_evans2012_noise),
            rhs_func(_rhs_func)
            {
        		assert(M.size()==3*T.size());
        		length=M.size();
        		dm_c= new double[length];
        		dm_pred= new double[length];
        		dw_t= new double[length];
        		dw_l= new double[length];

        		for (int i = 0; i < length; i++){
        			dw_t[i]=0;
        			dw_l[i]=0;
        		}

        		initial_random();

        		if (_rhs_func.is_none()) throw std::invalid_argument("HeunStochasticIntegrator::HeunStochasticIntegrator: _rhs_func is None");

         }

        ~HeunStochasticIntegrator(){


        	if (dm_pred!=0){
        		delete[] dm_pred;
        	}

        	if (dm_c!=0){
        	    delete[] dm_c;
        	}

        	if (dw_t!=0){
        	    delete[] dw_t;
        	}

        	if (dw_l!=0){
        	    delete[] dw_l;
        	}

        }

    	void run_step(const np_array<double> &H) {
    		double *h = H.data();
    		double *m = M.data();
    		double *m_pred=M_pred.data();

    		gauss_random_vec(dw_t,length,sqrt(dt));
    		gauss_random_vec(dw_l,length,sqrt(dt));

            bp::call<void>(rhs_func.ptr(),M);

    		calc_llb_adt_plus_bdw(m,h,dm_pred);

    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] + dm_pred[i];
    		}

    		bp::call<void>(rhs_func.ptr(), M_pred);

    		calc_llb_adt_plus_bdw(m_pred,h,dm_c);

    		for (int i = 0; i < length; i++){
    			m[i] += 0.5*(dm_c[i] + dm_pred[i]);
    		}

    	}



    private:
        int length;
        np_array<double> M,M_pred,T_arr,V_arr;
        double dt,gamma_LL,lambda,Tc,Ms;
        double *dm_pred, *dm_c,*dw_t, *dw_l;
        bool do_precession;
        bool use_evans2012_noise;

        bp::object rhs_func;


        //double cur_t, default_dt;

        void calc_llb_adt_plus_bdw(
        					double *m,
        					double *h,
        					double *dm
        					) {


                double precession_coeff = -gamma_LL;
                double *T = T_arr.data();
                double *V = V_arr.data();
                int len=T_arr.size();
                //printf("len=%d  length=%d\n",len,length);

                // calculate dm
                //#pragma omp parallel for schedule(guided)
                int i1,i2,i3;
                for (int i = 0; i < len; i++) {
                	i1=i;
                	i2=len+i1;
                	i3=len+i2;
                	//printf("%e  %e  %e  %e  %e  %e  %e\n",V[i],dw_l[i1],dw_l[i2],dw_l[i3],dw_t[i1],dw_t[i2],dw_t[i3]);

                    // add precession: m x H, multiplied by -gamma
                    if (do_precession) {
                        dm[i1] = precession_coeff*dt*(m[i2]*h[i3]-m[i3]*h[i2]);
                        dm[i2] = precession_coeff*dt*(m[i3]*h[i1]-m[i1]*h[i3]);
                        dm[i3] = precession_coeff*dt*(m[i1]*h[i2]-m[i2]*h[i1]);
                    } else {
                        dm[i1] = 0.;
                        dm[i2] = 0.;
                        dm[i3] = 0.;
                    }


                    // add transverse and longitudinal damping
                    double mm = m[i1] * m[i1] + m[i2] * m[i2] + m[i3] * m[i3];
                    double a_perp = alpha_perp(T[i], Tc);
                    double a_par = alpha_par(T[i], Tc);

                    // transverse damping noise
                    double b_tr = use_evans2012_noise ?
                        sqrt(
                            (2.*constant_K_B/constant_MU0)*T[i]*(a_perp - a_par) /
                            (gamma_LL * Ms * lambda * V[i] * a_perp*a_perp)
                        )
                    :
                        sqrt(
                            (2.*constant_K_B/constant_MU0)*T[i] /
                            (gamma_LL * Ms * lambda * V[i] * a_perp)
                        );

                    double h_tr_0 = h[i1]*dt + b_tr*dw_t[i1];
                    double h_tr_1 = h[i2]*dt + b_tr*dw_t[i2];
                    double h_tr_2 = h[i3]*dt + b_tr*dw_t[i3];

                    double mh_tr = m[i1] * h_tr_0 + m[i2] * h_tr_1 + m[i3] * h_tr_2;

                    // longitudinal damping noise
                    double b_long = use_evans2012_noise ?
                        sqrt(
                            (2.*gamma_LL*constant_K_B/constant_MU0*lambda)*T[i]*a_par /
                            (Ms * V[i])
                        )
                    :
                        sqrt(
                            (2.*constant_K_B/constant_MU0)*T[i] /
                            (gamma_LL * Ms * lambda * V[i] * a_par)
                        );

                    // add transverse damping: m x (m x H) == (m.H)m - (m.m)H, multiplied by -gamma lambda alpha_perp/m^2
                    double tdamp = -gamma_LL * lambda * a_perp / mm;
                    dm[i1] += tdamp * (mh_tr * m[i1] - mm * h_tr_0);
                    dm[i2] += tdamp * (mh_tr * m[i2] - mm * h_tr_1);
                    dm[i3] += tdamp * (mh_tr * m[i3] - mm * h_tr_2);

                    double mh_long;
                    if (use_evans2012_noise) {
                        double h_long_0 = h[i1]*dt;
                        double h_long_1 = h[i2]*dt;
                        double h_long_2 = h[i3]*dt;
                        mh_long = m[i1] * h_long_0 + m[i2] * h_long_1 + m[i3] * h_long_2;

                        // add noise separately
                        dm[i1] += b_long * dw_l[i1];
                        dm[i2] += b_long * dw_l[i2];
                        dm[i3] += b_long * dw_l[i3];

                    } else {
                        double h_long_0 = h[i1]*dt + b_long * dw_l[i1];
                        double h_long_1 = h[i2]*dt + b_long * dw_l[i2];
                        double h_long_2 = h[i3]*dt + b_long * dw_l[i3];
                        mh_long = m[i1] * h_long_0 + m[i2] * h_long_1 + m[i3] * h_long_2;
                    }


                    // add longitudinal damping: (m.H)m, muliplied by gamma lambda alpha_par/m^2
                    double ldamp = gamma_LL * lambda * a_par / mm;
                    dm[i1] += ldamp * mh_long * m[i1];
                    dm[i2] += ldamp * mh_long * m[i2];
                    dm[i3] += ldamp * mh_long * m[i3];
                    //printf("dm0==%e  dm1==%e  dm2==%e  mh_long==%e\n",dm[i1],dm[i2],dm[i3],mh_long);

                }
            }

    };



    class RungeKuttaStochasticIntegrator {
    public:

    	RungeKuttaStochasticIntegrator(
            const np_array<double> &M,
            const np_array<double> &M_pred,
            const np_array<double> &T,
            const np_array<double> &V,
            double dt,
            double gamma_LL,
            double lambda,
            double Tc,
            double Ms,
            bool do_precession,
            bp::object _rhs_func):
            M(M),
            M_pred(M_pred),
            T_arr(T),
            V_arr(V),
            dt(dt),
            gamma_LL(gamma_LL),
            lambda(lambda),
            Tc(Tc),
            Ms(Ms),
            do_precession(do_precession),
            rhs_func(_rhs_func)
            {
        		assert(M.size()==3*T.size());
        		length=M.size();
        		dm_c= new double[length];
        		dm_pred= new double[length];
        		eta_perp= new double[length];
        		eta_par= new double[length];

        		for (int i = 0; i < length; i++){
        			eta_perp[i]=0;
        			eta_par[i]=0;
        		}

        		initial_random();

        		if (_rhs_func.is_none()) throw std::invalid_argument("RungeKuttaStochasticIntegrator::RungeKuttaStochasticIntegrator: _rhs_func is None");

         }

        ~RungeKuttaStochasticIntegrator(){


        	if (dm_pred!=0){
        		delete[] dm_pred;
        	}

        	if (dm_c!=0){
        	    delete[] dm_c;
        	}

        	if (eta_perp!=0){
        	    delete[] eta_perp;
        	}

        	if (eta_par!=0){
        	    delete[] eta_par;
        	}

        }

    	void run_step(const np_array<double> &H) {
    		double *h = H.data();
    		double *m = M.data();
    		double *m_pred=M_pred.data();

    		gauss_random_vec(eta_perp,length,sqrt(dt));
    		gauss_random_vec(eta_par,length,sqrt(dt));

		bp::call<void>(rhs_func.ptr(),M);

    		calc_llb_adt_plus_bdw(m,h,dm_pred);

    		for (int i = 0; i < length; i++){
    			m_pred[i] = m[i] + 0.666666666666*dm_pred[i];
    		}

    		gauss_random_vec(eta_perp,length,sqrt(dt));
    		gauss_random_vec(eta_par,length,sqrt(dt));

    		bp::call<void>(rhs_func.ptr(), M_pred);

    		calc_llb_adt_plus_bdw(m_pred,h,dm_c);

    		for (int i = 0; i < length; i++){
    			m[i] += 0.25*dm_c[i] + 0.75*dm_pred[i];
    		}

    	}



    private:
        int length;
        np_array<double> M,M_pred,T_arr,V_arr;
        double dt,gamma_LL,lambda,Tc,Ms;
        double *dm_pred, *dm_c,*eta_perp, *eta_par;
        bool do_precession;

        bp::object rhs_func;


        //double cur_t, default_dt;

        void calc_llb_adt_plus_bdw(
        					double *m,
        					double *h,
        					double *dm
        					) {


                double *T = T_arr.data();
                double *V = V_arr.data();
                int len=T_arr.size();

                double precession_coeff = -gamma_LL;
                double damping_coeff = gamma_LL*lambda;

                // calculate dm
                int i1,i2,i3;
                for (int i = 0; i < len; i++) {
                	i1=i;
                	i2=len+i1;
                	i3=len+i2;

                    // add precession: m x H, multiplied by -gamma
                    if (do_precession) {
                        dm[i1] = precession_coeff*dt*(m[i2]*h[i3]-m[i3]*h[i2]);
                        dm[i2] = precession_coeff*dt*(m[i3]*h[i1]-m[i1]*h[i3]);
                        dm[i3] = precession_coeff*dt*(m[i1]*h[i2]-m[i2]*h[i1]);
                    } else {
                        dm[i1] = 0.;
                        dm[i2] = 0.;
                        dm[i3] = 0.;
                    }


                    double mh = m[i1] * h[i1] + m[i2] * h[i2] + m[i3] * h[i3];
                    double mm = m[i1] * m[i1] + m[i2] * m[i2] + m[i3] * m[i3];

                    //m x (m x H) == (m.H)m - (m.m)H,
                    double a_perp = alpha_perp(T[i1], Tc);
                    double a_par = alpha_par(T[i1], Tc);
                    double damp1 = (a_par-a_perp) * damping_coeff / mm * mh;
                    double damp2 = a_perp * damping_coeff;
                    dm[i1] += (m[i1] * damp1 + h[i1] * damp2)*dt;
                    dm[i2] += (m[i2] * damp1 + h[i2] * damp2)*dt;
                    dm[i3] += (m[i3] * damp1 + h[i3] * damp2)*dt;



                    double Q_perp =  sqrt(
                            (2.*constant_K_B)*T[i]*(a_perp - a_par) /
                            (gamma_LL * Ms* constant_MU0* V[i]* a_perp * a_perp * lambda)
                        );

                    double Q_par =sqrt(
                            (2.*gamma_LL*constant_K_B/lambda)*T[i]*a_par /
                            (Ms * V[i]*constant_MU0)
                        );
                    Q_par*=0.1;

                    double meta = m[i1] * eta_perp[i1] + m[i2] * eta_perp[i2] + m[i3] * eta_perp[i3];
                    //m x (m x H) == (m.H)m - (m.m)H,
                    double damp3 = -a_perp * damping_coeff / mm * meta;
                    dm[i1] += (m[i1] * damp3 + eta_perp[i1] * damp2)*Q_perp;
                    dm[i2] += (m[i2] * damp3 + eta_perp[i2] * damp2)*Q_perp;
                    dm[i3] += (m[i3] * damp3 + eta_perp[i3] * damp2)*Q_perp;

                    dm[i1] += Q_par*eta_par[i1];
                    dm[i2] += Q_par*eta_par[i2];
                    dm[i3] += Q_par*eta_par[i3];
                }
            }

    };

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



    	def("test_numpy",&test_numpy,(
    			arg("m")
    			));

    	class_<HeunStochasticIntegrator>("HeunStochasticIntegrator", init<
    			 	np_array<double>,
    			 	np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    double,
    			    double,
    			    double,
    			    double,
    			    double,
    			    bool,
    			    bool,
    			    bp::object>())
    	        	.def("run_step", &HeunStochasticIntegrator::run_step);

    	class_<RungeKuttaStochasticIntegrator>("RungeKuttaStochasticIntegrator", init<
    			 	np_array<double>,
    			 	np_array<double>,
    			    np_array<double>,
    			    np_array<double>,
    			    double,
    			    double,
    			    double,
    			    double,
    			    double,
    			    bool,
    			    bp::object>())
    	        	.def("run_step", &RungeKuttaStochasticIntegrator::run_step);
    }


}}
