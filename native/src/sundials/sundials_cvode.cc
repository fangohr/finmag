/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#include "finmag_includes.h"

#include <cvode/cvode_direct.h>

#include "sundials_cvode.h"

namespace finmag { namespace sundials {

    namespace {
        // Error message handler function
        void error_handler(int error_code, const char *module, const char *function, char *msg, void *eh_data);

        // Callbacks
        // ODE right-hand side
        int rhs_callback(realtype t, N_Vector y, N_Vector ydot, void *user_data);

        // Jacobian information (direct method with dense Jacobian)
        int dls_dense_jac_callback(int n, realtype t, N_Vector y, N_Vector fy, DlsMat Jac,
                    void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

        // Jacobian information (direct method with banded Jacobian)
        int band_jac_callback(int n, int mupper, int mlower, realtype t, N_Vector y, N_Vector fy,
                    DlsMat Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

        // Jacobian information (matrix-vector product)
        int spils_jac_times_vec_callback(N_Vector v, N_Vector Jv, realtype t, N_Vector y,
                    N_Vector fy, void *user_data, N_Vector tmp);

        // Preconditioning (linear system solution)
        int spils_prec_solve_callback(realtype t, N_Vector y, N_Vector fy, N_Vector r, N_Vector z,
                    realtype gamma, realtype delta, int lr, void *user_data, N_Vector tmp);

        // Preconditioning (Jacobian data)
        int spils_prec_setup_callback(realtype t, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr,
                    realtype gamma, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);
    }
}}