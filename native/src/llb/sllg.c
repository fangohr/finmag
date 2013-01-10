#include "sllg.h"
#include "llb_random.h"

ode_solver *create_ode_plan() {

	ode_solver *plan = (ode_solver*) malloc(sizeof(ode_solver));

	return plan;
}

void llg_rhs_dw(ode_solver *s, double *m, double *h, double *dm) {

	int i, j, k;

	double mth0, mth1, mth2;

	double mm, relax;

	int nxyz = s->nxyz;
	double *eta = s->eta;
	double dt = s->dt;
	double gamma = s->gamma;
	double coeff;
	double *alpha = s->alpha;
	double *T = s->T;
	double *V = s->V;
	double *Ms = s->Ms;
	double Q = s->Q;
	double q;

	gauss_random_vec(eta, 3 * s->nxyz, sqrt(dt));

	for (i = 0; i < nxyz; i++) {
		j = i + nxyz;
		k = j + nxyz;

		coeff = -gamma / (1.0 + alpha[i] * alpha[i]);
		q = sqrt(
				2 * Q * alpha[i] / (1.0 + alpha[i] * alpha[i]) * T[i] / (Ms[i]
						* V[i]));

		//printf("%g  %g   ",eta[i],dt);

		mth0 = coeff * (m[j] * h[k] - m[k] * h[j]) * dt;
		mth1 = coeff * (m[k] * h[i] - m[i] * h[k]) * dt;
		mth2 = coeff * (m[i] * h[j] - m[j] * h[i]) * dt;

		mth0 += coeff * (m[j] * eta[k] - m[k] * eta[j]) * q;
		mth1 += coeff * (m[k] * eta[i] - m[i] * eta[k]) * q;
		mth2 += coeff * (m[i] * eta[j] - m[j] * eta[i]) * q;

		dm[i] = mth0 + alpha[i] * (m[j] * mth2 - m[k] * mth1);
		dm[j] = mth1 + alpha[i] * (m[k] * mth0 - m[i] * mth2);
		dm[k] = mth2 + alpha[i] * (m[i] * mth1 - m[j] * mth0);

		/*
		 mm = m[i] * m[i] + m[j] * m[j] + m[k] * m[k];
		 relax = s->c * (1 - mm);
		 dm[i] += relax * m[i] * dt;
		 dm[j] += relax * m[j] * dt;
		 dm[k] += relax * m[k] * dt;
	    */

	}
	//printf("q\n\n");
}

void init_solver(ode_solver *s, double *alpha, double *T, double *V,
		double *Ms, int nxyz) {

	s->theta = 2.0 / 3.0;
	s->theta1 = 1.0 - 0.5 / s->theta;
	s->theta2 = 0.5 / s->theta;

	s->alpha = alpha;
	s->T = T;
	s->V = V;
	s->Ms = Ms;

	s->nxyz = nxyz;

	s->dm1 = (double*) malloc(3 * nxyz * sizeof(double));
	s->dm2 = (double*) malloc(3 * nxyz * sizeof(double));
	s->eta = (double*) malloc(3 * nxyz * sizeof(double));

	int i = 0;
	for (i = 0; i < 3 * nxyz; i++) {
		s->dm1[i] = 0;
		s->dm2[i] = 0;
		s->eta[i] = 0;
	}

	//initial_random();
}

void init_solver_parameters(ode_solver *s, double gamma, double dt, unsigned int seed) {

	double k_B = 1.3806505e-23;
	double mu_0 = 4 * M_PI * 1e-7;

	s->gamma = gamma;
	s->dt = dt;
	s->c=1e11;//delete later...

	s->Q = k_B / (gamma * mu_0);
	initial_random_with_seed(seed);
}

void run_step1(ode_solver *s, double *m, double *h, double *m_pred) {
	int i;
	double *dm1 = s->dm1;
	double theta = s->theta;

	llg_rhs_dw(s, m, h, dm1);

	for (i = 0; i < 3 * s->nxyz; i++) {
		m_pred[i] = m[i] + theta * dm1[i];
	}

}

double run_step2(ode_solver *s, double *m_pred, double *h, double *m) {
	int i, j, k;
	int nxyz = s->nxyz;
	double *dm1 = s->dm1;
	double *dm2 = s->dm2;
	double theta1 = s->theta1;
	double theta2 = s->theta2;

	llg_rhs_dw(s, m_pred, h, dm2);

	for (i = 0; i < 3 * nxyz; i++) {
		m[i] += (theta1 * dm1[i] + theta2 * dm2[i]);
	}

	double max_m=0;


	double mm;
	for (i = 0; i < s->nxyz; i++) {
		j = i + nxyz;
		k = j + nxyz;
		mm = sqrt(m[i] * m[i] + m[j] * m[j] + m[k] * m[k]);
		if (mm>max_m){
			max_m=mm;
		}
		mm=1.0/mm;
		m[i] *= mm;
		m[j] *= mm;
		m[k] *= mm;
	}

	return max_m;

}

double check_m(ode_solver *s, double *m) {
	double mm;
	int i,j,k;
	int nxyz = s->nxyz;
	double max_m=0;

	for (i = 0; i < nxyz; i++) {
		j = i + nxyz;
		k = j + nxyz;
		mm = sqrt(m[i] * m[i] + m[j] * m[j] + m[k] * m[k]);
		if (mm>max_m){
			max_m=mm;
		}
	}

	return max_m;
}

void finalize_ode_plan(ode_solver *plan) {
	free(plan->dm1);
	free(plan->dm2);
	free(plan->eta);
	free(plan);
}
