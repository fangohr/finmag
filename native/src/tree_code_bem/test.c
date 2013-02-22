#include "treecode_bem_helper.h"

//gcc test.c treecode_bem_helper.c -lm

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

    return 0;

}
