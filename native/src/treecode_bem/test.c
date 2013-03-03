#include "common.h"
#include <assert.h>

//gcc test.c common.c -lm

void test_compute_coefficient_directly(){
	double a[35];
	//computed using Mathematica
	double expected[35]={
			0.4800153607373193, 0.14378340298583298, 0.018603666107702595,
			-0.10465663185328682, -0.2580203559525883, 0.13272314121769196,
			0.11926733888225773, 0.08688209195428556, -0.037675500984807146,
			-0.0005096894824028226, 0.0654962728884902, 0.14649121265269124,
			-0.12303011561409782, -0.09308064949187649, -0.23706584465824798,
			0.12166287944955098, 0.10932839397540293, 0.07964191762476179,
			-0.03453587590273988, 0.10091851751575653, 0.1511452451272851,
			0.20065187628560183, 0.055419923213337864, 0.16658481317792173,
			-0.07876054957004934, -0.01809397662529978, 0.03916035896479662,
			0.11152914329989704, 0.036148023659812255, 0.13075615047668365,
			0.0905746320055567, -0.13506184083809963, -0.13204893727518185,
			-0.12189132671555247, -0.2021037753054538};

	double eps=1e-16;
	compute_coefficient_directly(a,1.1,1.2,1.3,4);


	int i;
	for(i=0;i<35;i++){
	   printf("a[%d]=%0.15g, diff=%g\n",i,a[i],a[i]-expected[i]);
	   assert(fabs(a[i]-expected[i])<eps);
	}


}



void test_single_layer_potential(int p){

	double ***a = alloc_3d_double(p + 1, p + 1, p + 1);
	double ***moment = alloc_3d_double(p + 1, p + 1, p + 1);

	double x[3]={2,0.2,0.3};
	double y[3]={0.12,0.13,0.14};
	double charge_density=1000;

	double dx=y[0];
	double dy=y[1];
	double dz=y[2];

	double tmp_x, tmp_y, tmp_z, R;
	double res_direct,res=0;

	int i,j,k;


	for (i = 0; i < p + 1; i++) {
	        for (j = 0; j < p - i + 1; j++) {
	            for (k = 0; k < p - i - j + 1; k++) {
	                moment[i][j][k] = 0;
	            }
	        }
	 }

    tmp_x = 1.0;
    for (i = 0; i < p + 1; i++) {
         tmp_y = 1.0;
         for (j = 0; j < p - i + 1; j++) {
                tmp_z = 1.0;
                for (k = 0; k < p - i - j + 1; k++) {
                    moment[i][j][k] += charge_density *tmp_x * tmp_y * tmp_z;
                    tmp_z *= dz;
                }
                tmp_y *= dy;
            }
            tmp_x *= dx;
     	 }

    compute_coefficient(a,x[0],x[1],x[2],p);
    for (i = 0; i < p + 1; i++) {
        for (j = 0; j < p - i + 1; j++) {
            for (k = 0; k < p - i - j + 1; k++) {
                res += a[i][j][k] * moment[i][j][k];
            }
        }
    }


    /*
     * compute directly
     */

    dx = x[0] - y[0];
    dy = x[1] - y[1];
    dz = x[2] - y[2];
    R = dx * dx + dy * dy + dz*dz;


    res_direct = charge_density / sqrt(R);

    printf("exact result: %g   fast sum: %g  and p=%d  rel_error=%g\n",res_direct,res,p,(res-res_direct)/res);

}



void test_double_layer_potential_I(int p){
	/*
	 * suppose yc=(0,0,0), triangle area A = 1, normal of triangle = norm(1,2,3)
	 * the middle point of the triangle is y=(0.12,0.13,0.14) and the charge density equals 1.
	 * the observing point is x = (11,0.2,0.3)
	 */

	double ***a = alloc_3d_double(p + 1, p + 1, p + 1);
	double ***moment = alloc_3d_double(p + 1, p + 1, p + 1);

	double x[3]={2,0.2,0.3};
	double y[3]={0.12,0.13,0.14};
	double n[3]={1,2,3};
	double charge_density=1000;

	double dx=y[0];
	double dy=y[1];
	double dz=y[2];

	double tmp_x, tmp_y, tmp_z, R;
	double res_direct,res=0;

	int i,j,k;

	vector_unit(n,n);

	for (i = 0; i < p + 1; i++) {
	        for (j = 0; j < p - i + 1; j++) {
	            for (k = 0; k < p - i - j + 1; k++) {
	                moment[i][j][k] = 0;
	            }
	        }
	 }

    tmp_x = 1.0;
    for (i = 0; i < p + 1; i++) {
         tmp_y = 1.0;
         for (j = 0; j < p - i + 1; j++) {
                tmp_z = 1.0;
                for (k = 0; k < p - i - j + 1; k++) {
                    moment[i][j][k] += charge_density * (
                         i * tmp_x * tmp_y * tmp_z / dx * n[0] +
                         j * tmp_x * tmp_y * tmp_z / dy * n[1] +
                         k * tmp_x * tmp_y * tmp_z / dz * n[2]);
                    tmp_z *= dz;
                }
                tmp_y *= dy;
            }
            tmp_x *= dx;
     	 }

    compute_coefficient(a,x[0],x[1],x[2],p);
    for (i = 0; i < p + 1; i++) {
        for (j = 0; j < p - i + 1; j++) {
            for (k = 0; k < p - i - j + 1; k++) {
                res += a[i][j][k] * moment[i][j][k];
            }
        }
    }


    /*
     * compute directly
     */

    dx = x[0] - y[0];
    dy = x[1] - y[1];
    dz = x[2] - y[2];
    R = dx * dx + dy * dy + dz*dz;


    dx *= n[0];
    dy *= n[1];
    dz *= n[2];

    res_direct = charge_density*(dx + dy + dz) / (R*sqrt(R));

    printf("exact result: %g   fast sum: %g  and p=%d  rel_error=%g\n",res_direct,res,p,(res-res_direct)/res);

}


int main() {

	int i;

	printf("Test single_layer_potential :\n");
	for (i=1;i<8;i++){
		test_single_layer_potential(i);
	}

	printf("Test double_layer_potential_I :\n");
	for (i=1;i<8;i++){
		test_double_layer_potential_I(i);
	}

	test_compute_coefficient_directly();

    return 0;

}
