#include "finmag_includes.h"
#include "util/np_array.h"
#include <vector>

namespace finmag { namespace llg {

    namespace {

    	double const const_pi = 3.1415926535897932384626433832795;
    	double const const_mu_0 = const_pi * 4e-7;

    	void compute_cubic_field(const np_array<double> &m,
    						const np_array<double> &Ms_arr,
            				const np_array<double> &H,
            				const np_array<double> &uv,
            				const np_array<double> &K1_arr,
            				const np_array<double> &K2_arr,
            				const np_array<double> &K3_arr){

                    m.check_ndim(2, "compute_cubic_field: m");
                    int const nodes = m.dim()[1];

                    m.check_shape(3, nodes, "compute_cubic_field: m");
                    H.check_shape(3, nodes, "compute_cubic_field: H");

                    Ms_arr.check_shape(nodes, "compute_cubic_field: Ms");
                    uv.check_shape(9, "compute_cubic_field: uv");

                    double *mx = m(0), *my = m(1), *mz = m(2);
                    double *hx = H(0), *hy = H(1), *hz = H(2);

                    double u1_x=(*uv[0]), u1_y=(*uv[1]), u1_z=(*uv[2]);
                    double u2_x=(*uv[3]), u2_y=(*uv[4]), u2_z=(*uv[5]);
                    double u3_x=(*uv[6]), u3_y=(*uv[7]), u3_z=(*uv[8]);

                    double *Ms = Ms_arr.data();
                    double *K1 = K1_arr.data();
                    double *K2 = K2_arr.data();
                    double *K3 = K3_arr.data();

					//#pragma omp parallel for schedule(guided)
                    for (int i=0; i < nodes; i++) {
                    	double u1m=mx[i]*u1_x+my[i]*u1_y+mz[i]*u1_z;
                    	double u2m=mx[i]*u2_x+my[i]*u2_y+mz[i]*u2_z;
                    	double u3m=mx[i]*u3_x+my[i]*u3_y+mz[i]*u3_z;

                    	double u1m_sq=u1m*u1m;
                    	double u2m_sq=u2m*u2m;
                    	double u3m_sq=u3m*u3m;

                    	hx[i]=0;
                    	hy[i]=0;
                    	hz[i]=0;
                    	double t1,t2,t3;

                    	if (K1[i]!=0){
                    		t1=(u2m_sq+u3m_sq)*u1m;
                    		t2=(u1m_sq+u3m_sq)*u2m;
                    		t3=(u1m_sq+u2m_sq)*u3m;

                    		hx[i] += K1[i]*(t1*u1_x+t2*u2_x+t3*u3_x);
                    		hy[i] += K1[i]*(t1*u1_y+t2*u2_y+t3*u3_y);
                    		hz[i] += K1[i]*(t1*u1_z+t2*u2_z+t3*u3_z);
                    	}

                    	if (K2[i]!=0){
                    		t1=(u2m_sq*u3m_sq)*u1m;
                    		t2=(u1m_sq*u3m_sq)*u2m;
                    		t3=(u1m_sq*u2m_sq)*u3m;

                    		hx[i] += K2[i]*(t1*u1_x+t2*u2_x+t3*u3_x);
                    		hy[i] += K2[i]*(t1*u1_y+t2*u2_y+t3*u3_y);
                    		hz[i] += K2[2]*(t1*u1_z+t2*u2_z+t3*u3_z);
                    	}

                    	if (K3[i]!=0){
                    		t1=(u2m_sq*u2m_sq+u3m_sq*u3m_sq)*u1m_sq*u1m;
                    		t2=(u1m_sq*u1m_sq+u3m_sq*u3m_sq)*u2m_sq*u2m;
                    		t3=(u1m_sq*u1m_sq+u2m_sq*u2m_sq)*u3m_sq*u3m;

                    		hx[i] += 2*K3[i]*(t1*u1_x+t2*u2_x+t3*u3_x);
                    		hy[i] += 2*K3[i]*(t1*u1_y+t2*u2_y+t3*u3_y);
                    		hz[i] += 2*K3[i]*(t1*u1_z+t2*u2_z+t3*u3_z);
                    	}

                    	if (Ms[i]!=0){
                    		t1=-2.0/(const_mu_0*Ms[i]);
                    		hx[i]*=t1;
                    		hy[i]*=t1;
                    		hz[i]*=t1;
                    	}else{
                    		hx[i]=0;
                    		hy[i]=0;
                    		hz[i]=0;
                    	}


                    }

    	}



    }

    void register_energy() {
        using namespace bp;

        def("compute_cubic_field", &compute_cubic_field, (
            arg("m"),
            arg("Ms_arr"),
            arg("H"),
            arg("uv"),
            arg("K1_arr"),
            arg("K2_arr"),
            arg("K3_arr")
        ));

    }

}}



