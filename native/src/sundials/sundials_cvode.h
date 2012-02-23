/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#ifndef __FINMAG_ODE_SUNDIALS_CVODE_H
#define __FINMAG_ODE_SUNDIALS_CVODE_H

namespace finmag { namespace sundials {

    class cvode {
    public:
        cvode(int lmm, int iter): cvode_mem(0) {
            if (lmm != CV_ADAMS && lmm != CV_BDF)
                throw std::invalid_argument("sundials_cvode: lmm parameter must be either CV_ADAMS or CV_BDF");
            if (iter != CV_NEWTON && iter != CV_FUNCTIONAL)
                throw std::invalid_argument("sundials_cvode: iter parameter must be either CV_NEWTON or CV_FUNCTIONAL");
            cvode_mem = CVodeCreate(lmm, iter);
            if (!cvode_mem) throw std::runtime_error("CVodeCreate returned NULL");
        }

        ~cvode() {
            if (cvode_mem) {
                CVodeFree(&cvode_mem);
                cvode_mem = 0;
            }
        }

        // initialisation functions
        void init(const bp::object &f, double t0, const np_array<double>& vec) {

        }

        void set_scalar_tolerances(double reltol, double abstol) {

        }

        // linear soiver specification functions
        void set_linear_solver_dense(int n) {}

        void set_linear_solver_lapack_dense(int n) {}

        void set_linear_solver_band(int n, int mupper, int mlower) {}

        void set_linear_solver_lapack_band(int n, int mupper, int mlower) {}

        void set_linear_solver_sp_gmr(int pretype, int maxl) {}

        void set_linear_solver_sp_bcg(int pretype, int maxl) {}

        void set_linear_solver_sp_tfqmr(int pretype, int maxl) {}

        // solver functions
        double advance_time(double tout, const np_array<double> &yout, int itask) { return 0.; }

        // main solver optional input functions
        void set_max_ord(int max_order) {}

        void set_max_num_steps(int max_steps) {}

        void set_max_hnil_warns(int max_hnil) {}

        void set_stab_lim_det(bool stldet) {}

        void set_init_step(double initial_step_size) {}

        void set_min_step_size(double min_step_size) {}

        void set_max_step_size(double max_step_size) {}

        void set_stop_time(double stop_time) {}

        void set_max_err_test_fails(int max_err_test_fails) {}

        void set_max_nonlin_iters(int max_nonlin_iters) {}

        void set_max_conv_fails(int max_conv_fails) {}

        void set_nonlin_conv_coef(double nonlin_conv_coef) {}

        void set_iter_type(int iter_type) {}

        // direct linear solver optional input functions

        void set_dls_jac_fn(const bp::object &djac) {}

        void set_dls_band_jac_fn(const bp::object &bjac) {}

        // iterative linear solver optional input functions

        void set_spils_preconditioner(const bp::object &psetup, const bp::object &psolve) {}

        void set_splis_jac_times_vec_fn(const bp::object &jtimes) {}

        void set_spils_prec_type(int pretype) {}

        void set_spils_gs_type(int gstype) {}

        void set_spils_eps_lin(double eplifac) {}

        void set_spils_maxl(int maxl) {}

        // TODO: add rootfinding and interpolation methods

        // optional output functions

        bp::tuple get_work_space() { return bp::tuple(); }

        int get_num_steps() { return 0; }

        int get_num_rhs_evals() { return 0; }

        int get_num_lin_solv_setups() { return 0; }

        int get_num_err_test_fails() { return 0; }

        int get_last_order() { return 0; }

        int get_current_order() { return 0; }

        double get_last_step() { return 0; }

        double get_current_step() { return 0; }

        double get_actual_init_step() { return 0; }

        double get_current_time() { return 0; }

        long get_num_stab_lim_order_reds() { return 0; }

        double get_tol_scale_factor() { return 0; }

        void get_err_weights(np_array<double> &arr) {}

        void get_est_local_errors(np_array<double> &arr) {}

        bp::tuple get_integrator_stats() { return bp::tuple(); }

        int get_num_nonlin_solv_iters() { return 0; }

        int get_num_nonlin_solv_conv_fails() { return 0; }

        bp::tuple get_nonlin_solv_stats() { return bp::tuple(); }

        std::string get_return_flag_name(int flag) { return ""; }

        // direct linear solver optional output functions

        bp::tuple get_dls_work_space() { return bp::tuple(); }

        int get_dls_num_jac_evals() { return 0; }

        int get_dls_num_rhs_evals() { return 0; }

        int get_dls_last_flag() { return 0; }

        std::string get_dls_return_flag_name(int flag) { return ""; }

        // diagonal linear solver optional output functions

        bp::tuple get_diag_work_space() { return bp::tuple(); }

        int get_diag_num_rhs_evals() { return 0; }

        int get_diag_last_flag() { return 0; }

        std::string get_diag_return_flag_name() { return ""; }

        // iterative linear solver optional output functions

        bp::tuple get_spils_work_space() { return bp::tuple(); }

        int get_spils_num_lin_iters() { return 0; }

        int get_spils_num_conv_fails() { return 0; }

        int get_spils_num_prec_evals() { return 0; }

        int get_spils_num_prec_solves() { return 0; }

        int get_spils_num_jtimes_evals() { return 0; }

        int get_spils_num_rhs_evals() { return 0; }

        int get_spils_last_flag() { return 0; }

        std::string get_spils_return_flag_name(int flag) { return ""; }

        // reinitialisation functions

        void reinit(double t0, const np_array<double> &y0) {}

        // preconditioner functions

        void band_prec_init(int n, int mu, int ml) {}

        bp::tuple get_band_prec_work_space() { return bp::tuple(); }

        int get_band_prec_num_rhs_evals() { return 0; }


    private:
        void* cvode_mem;

        bp::object rhs_fn, dls_jac_fn, dls_band_jac_fn, spils_prec_setup_fn, spils_prec_solve_fn, spils_jac_times_vec_fn;
    };

    void register_sundials_cvode() {
        using namespace bp;

        class_<cvode>("sundials_cvode", init<int, int>(args("lmm", "iter")))
        ;

        scope().attr("CV_ADAMS") = int(CV_ADAMS);
        scope().attr("CV_BDF") = int(CV_BDF);
        scope().attr("CV_NEWTON") = int(CV_NEWTON);
        scope().attr("CV_FUNCTIONAL") = int(CV_FUNCTIONAL);
    }
}}

#endif