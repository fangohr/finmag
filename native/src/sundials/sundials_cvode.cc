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
        void error_callback(int error_code, const char *module, const char *function, char *msg, void *eh_data) {
            cvode *cv = (cvode*) eh_data;

            cv->on_error(error_code, module, function, msg);
        }

        // Callbacks
        // ODE right-hand side
        int rhs_callback(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
            return 0;
        }

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

    cvode::cvode(int lmm, int iter): cvode_mem(0) {
        if (lmm != CV_ADAMS && lmm != CV_BDF)
            throw std::invalid_argument("sundials_cvode: lmm parameter must be either CV_ADAMS or CV_BDF");
        if (iter != CV_NEWTON && iter != CV_FUNCTIONAL)
            throw std::invalid_argument("sundials_cvode: iter parameter must be either CV_NEWTON or CV_FUNCTIONAL");
        cvode_mem = CVodeCreate(lmm, iter);
        if (!cvode_mem) throw std::runtime_error("CVodeCreate returned NULL");

        // save this object as CVODE user data
        int flag = CVodeSetUserData(cvode_mem, this);
        if (flag != CV_SUCCESS) {
            // this shouldn't happen...
            CVodeFree(&cvode_mem);
            cvode_mem = 0;
            throw std::runtime_error("CVodeSetUserData failed");
        }

        // set up the error handler
        flag = CVodeSetErrHandlerFn(cvode_mem, error_callback, this);
        if (flag != CV_SUCCESS) {
            // this shouldn't happen, either...
            CVodeFree(&cvode_mem);
            cvode_mem = 0;
            throw std::runtime_error("CVodeSetErrHandlerFn failed");
        }
    }

    void cvode::on_error(int error_code, const char *module, const char *function, char *msg) {
        fprintf(stderr, "Error in %s:%s (%d): %s", module, function, error_code, msg);
        // TODO: save the error
    }

    void register_sundials_cvode() {
        using namespace bp;

        class_<cvode>("sundials_cvode", init<int, int>(args("lmm", "iter")))
            // initialisation functions
            .def("init", &cvode::init, (arg("f"), arg("t0"), arg("y0")))
            .def("set_scalar_tolerances", &cvode::set_scalar_tolerances, (arg("reltol"), arg("abstol")))
            // linear soiver specification functions
            .def("set_linear_solver_dense", &cvode::set_linear_solver_dense, (arg("n")))
            .def("set_linear_solver_lapack_dense", &cvode::set_linear_solver_lapack_dense, (arg("n")))
            .def("set_linear_solver_band", &cvode::set_linear_solver_band, (arg("n"), arg("mupper"), arg("mlower")))
            .def("set_linear_solver_lapack_band", &cvode::set_linear_solver_lapack_band, (arg("n"), arg("mupper"), arg("mlower")))
            .def("set_linear_solver_sp_gmr", &cvode::set_linear_solver_sp_gmr, (arg("pretype"), arg("maxl")))
            .def("set_linear_solver_sp_bcg", &cvode::set_linear_solver_sp_bcg, (arg("pretype"), arg("maxl")))
            .def("set_linear_solver_sp_tfqmr", &cvode::set_linear_solver_sp_tfqmr, (arg("pretype"), arg("maxl")))
            // solver functions
            .def("advance_time", &cvode::advance_time, (arg("tout"), arg("yout"), arg("itask")=CV_NORMAL))
        ;

        scope().attr("CV_ADAMS") = int(CV_ADAMS);
        scope().attr("CV_BDF") = int(CV_BDF);
        scope().attr("CV_NEWTON") = int(CV_NEWTON);
        scope().attr("CV_FUNCTIONAL") = int(CV_FUNCTIONAL);
    }
}}