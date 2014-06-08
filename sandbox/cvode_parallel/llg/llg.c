#include "llg.h"

int llg_rhs(Vec M, Vec H, Vec dM_dt, Vec alpha_v,  double gamma, int do_precession, double char_freq) {
	
    PetscReal *m, *h, *dm_dt, *alpha;
    PetscReal mth0, mth1, mth2, a1, mm;
    int i,j,nlocal=0;
    VecGetArray(M, &m);
    VecGetArray(H, &h);
    VecGetArray(dM_dt, &dm_dt);
    VecGetArray(alpha_v, &alpha);
    
    VecGetLocalSize(M,&nlocal);
    
    if (do_precession) {
    	
        for (i = 0; i < nlocal; i += 3) {
        	
        	j = i/3;
        	
        	a1 = -gamma / (1 + alpha[j] * alpha[j]);
        	
            mth0 = a1 * (m[i + 1] * h[i + 2] - m[i + 2] * h[i + 1]);
            mth1 = a1 * (m[i + 2] * h[i] - m[i] * h[i + 2]);
            mth2 = a1 * (m[i] * h[i + 1] - m[i + 1] * h[i]);

            dm_dt[i] = mth0 + alpha[j] * (m[i + 1] * mth2 - m[i + 2] * mth1);
            dm_dt[i + 1] = mth1 + alpha[j] * (m[i + 2] * mth0 - m[i ] * mth2);
            dm_dt[i + 2] = mth2 + alpha[j] * (m[i] * mth1 - m[i + 1] * mth0);
            
            mm = m[i] * m[i] + m[i + 1] * m[i + 1] + m[i + 2] * m[i + 2];
            dm_dt[i] += char_freq*(1-mm)*m[i];
            dm_dt[i+1] += char_freq*(1-mm)*m[i+1];
            dm_dt[i+2] += char_freq*(1-mm)*m[i+2];
        }

    } else {

        for (i = 0; i < nlocal; i += 3) {
        	
        	j = i/3;
        	
        	a1 = -gamma / (1 + alpha[j] * alpha[j]);
        	
            mth0 = a1 * (m[i + 1] * h[i + 2] - m[i + 2] * h[i + 1]);
            mth1 = a1 * (m[i + 2] * h[i ] - m[i] * h[i + 2]);
            mth2 = a1 * (m[i] * h[i + 1] - m[i + 1] * h[i]);

            dm_dt[i] = alpha[j] * (m[i + 1] * mth2 - m[i + 2] * mth1);
            dm_dt[i + 1] = alpha[j] * (m[i + 2] * mth0 - m[i ] * mth2);
            dm_dt[i + 2] = alpha[j] * (m[i] * mth1 - m[i + 1] * mth0);
            
            mm = m[i] * m[i] + m[i + 1] * m[i + 1] + m[i + 2] * m[i + 2];
            dm_dt[i] += char_freq*(1-mm)*m[i];
            dm_dt[i+1] += char_freq*(1-mm)*m[i+1];
            dm_dt[i+2] += char_freq*(1-mm)*m[i+2];
        }

    }

    VecRestoreArray(M, &m);
    VecRestoreArray(H, &h);
    VecRestoreArray(dM_dt, &dm_dt);
    VecRestoreArray(alpha_v, &alpha);

    return 0;
}

