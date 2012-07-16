#include "finmag_includes.h"

#include "util/np_array.h"

#include "llb.h"


namespace finmag { namespace llb {
namespace {

	static const double constant_MU0 = M_PI*4e-7; // T m/A
    static const double constant_K_B = 1.3806488e-23; // J/K

    inline double cross0(double a0, double a1, double a2, double b0, double b1, double b2) { return a1*b2 - a2*b1; }
    inline double cross1(double a0, double a1, double a2, double b0, double b1, double b2) { return a2*b0 - a0*b2; }
    inline double cross2(double a0, double a1, double a2, double b0, double b1, double b2) { return a0*b1 - a1*b0; }


    inline double alpha_perp(double T, double T_C) { return T <= T_C ? 1. - (1./3.)*(T/T_C) : (2./3.) * (T/T_C); }
    inline double alpha_par(double T, double T_C) { return (2./3.) * (T/T_C); }

    void test_numpy(const np_array<double> &m){
    	printf("size of m:%d\n",m.size());
    	int length=m.size();
    	assert(length%3==0);
    	double *M=m.data();
    	for (int i=0;i<length;i++){
    		printf("m[%d]=%f\n",i,M[i]);
    		M[i]*=2;

    	}
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


    void calc_llb_adt_plus_bdw(
            const np_array<double> &M,
            const np_array<double> &H,
            const np_array<double> &T_arr,
            const np_array<double> &V_arr,
            const np_array<double> &dw_arr,
            const np_array<double> &dmdt,
            double dt,
            double gamma_LL,
            double lambda,
            double Tc,
            double Ms,
            bool do_precession,
            bool use_evans2012_noise) {


    	int length=M.size()/3;
    	double *m=M.data();
    	double *h=H.data();
    	double *dm_dt=dmdt.data();
        double *T = T_arr.data();
        double *V = V_arr.data();

        double *dw0 = dw_arr(0), *dw1 = dw_arr(1), *dw2 = dw_arr(2);
        double *dw3 = dw_arr(3), *dw4 = dw_arr(4), *dw5 = dw_arr(5);


        double precession_coeff = -gamma_LL;


        // calculate dm
        //#pragma omp parallel for schedule(guided)
        int i1,i2,i3;
        for (int i = 0; i < length; i++) {
        	i1=i;
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

            double h_tr_0 = h[i1]*dt + b_tr*dw0[i];
            double h_tr_1 = h[i2]*dt + b_tr*dw1[i];
            double h_tr_2 = h[i3]*dt + b_tr*dw2[i];
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
            dm_dt[i1] += tdamp * (mh_tr * m[i1] - mm * h_tr_0);
            dm_dt[i2] += tdamp * (mh_tr * m[i2] - mm * h_tr_1);
            dm_dt[i3] += tdamp * (mh_tr * m[i3] - mm * h_tr_2);

            double mh_long;
            if (use_evans2012_noise) {
                double h_long_0 = h[i1]*dt;
                double h_long_1 = h[i2]*dt;
                double h_long_2 = h[i3]*dt;
                mh_long = m[i1] * h_long_0 + m[i2] * h_long_1 + m[i3] * h_long_2;

                // add noise separately
                dm_dt[i1] += b_long * dw3[i];
                dm_dt[i2] += b_long * dw4[i];
                dm_dt[i3] += b_long * dw5[i];

            } else {
                double h_long_0 = h[i1]*dt + b_long * dw3[i];
                double h_long_1 = h[i2]*dt + b_long * dw4[i];
                double h_long_2 = h[i3]*dt + b_long * dw5[i];
                mh_long = m[i1] * h_long_0 + m[i2] * h_long_1 + m[i3] * h_long_2;
            }

            // add longitudinal damping: (m.H)m, muliplied by gamma lambda alpha_par/m^2
            double ldamp = gamma_LL * lambda * a_par / mm;
            dm_dt[i1] += ldamp * mh_long * m[i1];
            dm_dt[i2] += ldamp * mh_long * m[i2];
            dm_dt[i3] += ldamp * mh_long * m[i3];

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


    	def("calc_llb_adt_plus_bdw", &calc_llb_adt_plus_bdw, (
    	            arg("M"),
    	            arg("H"),
    	            arg("T_arr"),
    	            arg("V_arr"),
    	            arg("dw_arr"),
    	            arg("dmdt"),
    	            arg("dt"),
    	            arg("gamma_LL"),
    	            arg("lambda"),
    	            arg("Tc"),
    	            arg("Ms"),
    	            arg("do_precession"),
    	            arg("use_evans2012_noise")
    	        ));

    	def("test_numpy",&test_numpy,(
    			arg("m")
    			));

    }



}}
