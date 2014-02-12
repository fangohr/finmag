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

#include "util/np_array.h"
#include "numpy_malloc.h"
#include "util/python_threading.h"

#include <cvode/cvode_direct.h>
#include <cvode/cvode_dense.h>
#include <cvode/cvode_lapack.h>
#include <cvode/cvode_band.h>
#include <cvode/cvode_bandpre.h>
#include <cvode/cvode_diag.h>
#include <cvode/cvode_impl.h>
#include <cvode/cvode_spgmr.h>
#include <cvode/cvode_spbcgs.h>
#include <cvode/cvode_sptfqmr.h>

#define CHECK_SUNDIALS_RET(fn, args) do { \
        error_handler _eh; \
        int _retcode = fn args; \
        _eh.check_error(_retcode, #fn); \
    } while (0)

/*
    Currently, we support SUNDIALS versions 2.4 (Ubuntu 12.04) and 2.5 (Ubuntu 12.10)
    To make our life fun, SUNDIALS does not provide a get_version function, and its library files are not numbered, which makes checking
    the version at runtime quite difficult. If the wrong version is loaded at runtime, we might get seg faults or silent corruption.

    TODO: check that the correct sundials version is loaded at runtime

    Function callback signatures in 2.5 are different from 2.4: in 2.5, some int parameters have become long parameters.
    To get around this, we need to define a type that will be either int, or long, depending on the SUNDIALS version we're building
    against (in the SUNDIALS headers, the signature just changed from int to long without a common typedef that we could use).

    Unfortunately, to make our life more fun, SUNDIALS header do not define an integer version number that we could use
    in an #if condition.
*/
// First, convert the string SUNDIALS_PACKAGE_VERSION to a number
// The string has to have the form of "2.4.0" or "2.5.0"
// Using C++11 constexpr here is the easiest option, but there are (longer) workarounds if we are not allowed to use the new standard
constexpr bool const_str_equal(const char (&a)[6], const char (&b)[6]) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3] && a[4] == b[4] && a[5] == b[5];
}
constexpr int get_sundials_version_number(const char (&v)[6]) {
    return const_str_equal(v, "2.5.0") ? 250 : const_str_equal(v, "2.4.0") ? 240 : -1;
}

// Next, define the parameter type 'sundials_long_param_t' based on the version number
template<int Version> struct sundials_traits;
template<> struct sundials_traits<240> { typedef int param_t; };
template<> struct sundials_traits<250> { typedef long param_t; };
typedef typename sundials_traits<get_sundials_version_number(SUNDIALS_PACKAGE_VERSION)>::param_t sundials_long_param_t;

namespace finmag { namespace sundials {

    class error_handler {
    public:
        error_handler() {
            // We should never have an earlier unprocessed error message
            // If this does happen, we should just call cvode_error.reset()
            ASSERT(!cvode_error.get());
        }

        ~error_handler() {
            cvode_error.reset();
        }

        void check_error(int retcode, const char *function) {
            if (retcode >= 0) {
                // success error code, we should have no error messages
                ASSERT(!cvode_error.get());
            } else {
                // failure error code, we should have an error message
                std::auto_ptr<std::string> msg(cvode_error.release());
                if (msg.get()) {
                    throw std::runtime_error(*msg);
                } else {
                    throw std::runtime_error(std::string(function) + " has returned error " + boost::lexical_cast<std::string>(retcode));
                }
            }
        }

        static void set_error(const char*);

    private:
        error_handler(const error_handler&);
        void operator=(const error_handler&);

        static boost::thread_specific_ptr<std::string> cvode_error;
    };

    /* Note that there is also a function called get_sundials_version_number()
       defined above, but this doesn't seem to be used anywhere at the moment.
       Also, that function returns an integer (e.g. 240 for version 2.4.0)
       whereas here we return a string (such as "2.4.0").
       -- Max, 12.2.2014
     */
    std::string get_sundials_version() {
      return SUNDIALS_PACKAGE_VERSION;
    }

    class cvode {
    public:
        cvode(int lmm, int iter);

        ~cvode() {
            if (cvode_mem) {
                CVodeFree(&cvode_mem);
                cvode_mem = 0;
            }
        }

        // initialisation functions
        void init(const bp::object &f, double t0, const np_array<double>& y0);

        void set_scalar_tolerances(double reltol, double abstol);

        // linear solver specification functions
        void set_linear_solver_dense(int n) {
            CHECK_SUNDIALS_RET(CVDense, (cvode_mem, n));
        }

        void set_linear_solver_lapack_dense(int n) {
            CHECK_SUNDIALS_RET(CVLapackDense, (cvode_mem, n));
        }

        void set_linear_solver_band(int n, int mupper, int mlower) {
            CHECK_SUNDIALS_RET(CVBand, (cvode_mem, n, mupper, mlower));
        }

        void set_linear_solver_lapack_band(int n, int mupper, int mlower) {
            CHECK_SUNDIALS_RET(CVLapackBand, (cvode_mem, n, mupper, mlower));
        }

        void set_linear_solver_diag() {
            CHECK_SUNDIALS_RET(CVDiag, (cvode_mem));
        }

        void set_linear_solver_sp_gmr(int pretype, int maxl) {
            CHECK_SUNDIALS_RET(CVSpgmr, (cvode_mem, pretype, maxl));
        }

        void set_linear_solver_sp_bcg(int pretype, int maxl) {
            CHECK_SUNDIALS_RET(CVSpbcg, (cvode_mem, pretype, maxl));
        }

        void set_linear_solver_sp_tfqmr(int pretype, int maxl) {
            CHECK_SUNDIALS_RET(CVSptfqmr, (cvode_mem, pretype, maxl));
        }

        // solver functions
        double advance_time(double tout, const np_array<double> &yout, int itask);

        // main solver optional input functions
        void set_max_ord(int max_order) {
            CHECK_SUNDIALS_RET(CVodeSetMaxOrd, (cvode_mem, max_order));
        }

        void set_max_num_steps(int max_steps) {
            CHECK_SUNDIALS_RET(CVodeSetMaxNumSteps, (cvode_mem, max_steps));
        }

        void set_max_hnil_warns(int max_hnil) {
            CHECK_SUNDIALS_RET(CVodeSetMaxHnilWarns, (cvode_mem, max_hnil));
        }

        void set_stab_lim_det(bool stldet) {
            CHECK_SUNDIALS_RET(CVodeSetStabLimDet, (cvode_mem, stldet));
        }

        void set_init_step(double initial_step_size) {
            CHECK_SUNDIALS_RET(CVodeSetInitStep, (cvode_mem, initial_step_size));
        }

        void set_min_step_size(double min_step_size) {
            CHECK_SUNDIALS_RET(CVodeSetMinStep, (cvode_mem, min_step_size));
        }

        void set_max_step_size(double max_step_size) {
            CHECK_SUNDIALS_RET(CVodeSetMaxStep, (cvode_mem, max_step_size));
        }

        void set_stop_time(double stop_time) {
            CHECK_SUNDIALS_RET(CVodeSetStopTime, (cvode_mem, stop_time));
        }

        void set_max_err_test_fails(int max_err_test_fails) {
            CHECK_SUNDIALS_RET(CVodeSetMaxErrTestFails, (cvode_mem, max_err_test_fails));
        }

        void set_max_nonlin_iters(int max_nonlin_iters) {
            CHECK_SUNDIALS_RET(CVodeSetMaxNonlinIters, (cvode_mem, max_nonlin_iters));
        }

        void set_max_conv_fails(int max_conv_fails) {
            CHECK_SUNDIALS_RET(CVodeSetMaxConvFails, (cvode_mem, max_conv_fails));
        }

        void set_nonlin_conv_coef(double nonlin_conv_coef) {
            CHECK_SUNDIALS_RET(CVodeSetNonlinConvCoef, (cvode_mem, nonlin_conv_coef));
        }

        void set_iter_type(int iter_type) {
            CHECK_SUNDIALS_RET(CVodeSetIterType, (cvode_mem, iter_type));
        }

        // direct linear solver optional input functions

        void set_dls_jac_fn(const bp::object &djac) {
            dls_jac_fn = djac;
            CHECK_SUNDIALS_RET(CVDlsSetDenseJacFn, (cvode_mem, &dls_dense_jac_callback));
        }

        void set_dls_band_jac_fn(const bp::object &bjac) {
            dls_band_jac_fn = bjac;
            CHECK_SUNDIALS_RET(CVDlsSetBandJacFn, (cvode_mem, &dls_band_jac_callback));
        }

        // iterative linear solver optional input functions

        void set_spils_preconditioner(const bp::object &psetup, const bp::object &psolve) {
            spils_prec_setup_fn = psetup;
            spils_prec_solve_fn = psolve;
            CHECK_SUNDIALS_RET(CVSpilsSetPreconditioner, (cvode_mem, &spils_prec_setup_callback, &spils_prec_solve_callback));
        }

        void set_spils_jac_times_vec_fn(const bp::object &jtimes) {
            spils_jac_times_vec_fn = jtimes;
            CHECK_SUNDIALS_RET(CVSpilsSetJacTimesVecFn, (cvode_mem, &spils_jac_times_vec_callback));
        }

        void set_spils_prec_type(int pretype) {
            CHECK_SUNDIALS_RET(CVSpilsSetPrecType, (cvode_mem, pretype));
        }

        void set_spils_gs_type(int gstype) {
            CHECK_SUNDIALS_RET(CVSpilsSetGSType, (cvode_mem, gstype));
        }

        void set_spils_eps_lin(double eplifac) {
            CHECK_SUNDIALS_RET(CVSpilsSetEpsLin, (cvode_mem, eplifac));
        }

        void set_spils_maxl(int maxl) {
            CHECK_SUNDIALS_RET(CVSpilsSetMaxl, (cvode_mem, maxl));
        }

        // TODO: add rootfinding and interpolation methods

        // optional output functions

        bp::tuple get_work_space() {
            long lenrw = 0, leniw = 0;
            CHECK_SUNDIALS_RET(CVodeGetWorkSpace, (cvode_mem, &lenrw, &leniw));
            return bp::make_tuple(lenrw, leniw);
        }

        long get_num_steps() {
            long nsteps = 0;
            CHECK_SUNDIALS_RET(CVodeGetNumSteps, (cvode_mem, &nsteps));
            return nsteps;
        }

        long get_num_rhs_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetNumRhsEvals, (cvode_mem, &retval));
            return retval;
        }

        long get_num_lin_solv_setups() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetNumLinSolvSetups, (cvode_mem, &retval));
            return retval;
        }

        long get_num_err_test_fails() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetNumErrTestFails, (cvode_mem, &retval));
            return retval;
        }

        int get_last_order() {
            int retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetLastOrder, (cvode_mem, &retval));
            return retval;
        }

        int get_current_order() { 
            int retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetCurrentOrder, (cvode_mem, &retval));
            return retval;
        }

        double get_last_step() { 
            double retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetLastStep, (cvode_mem, &retval));
            return retval;
        }

        double get_current_step() { 
            double retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetCurrentStep, (cvode_mem, &retval));
            return retval;
        }

        double get_actual_init_step() { 
            double retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetActualInitStep, (cvode_mem, &retval));
            return retval;
        }

	/* Should be aware that get_current_time returns the INTERNAL time of the 
	   integrator, not the time up to which the user requested integration.
	   Any credit/blame for this choice of name should go to the sundials team -- what
	   we do here is simply exposing the C-functions to Python.

	   Dmitr, Hans, 12/12/12
	*/

        double get_current_time() { 
            double retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetCurrentTime, (cvode_mem, &retval));
            return retval;
        }

        long get_num_stab_lim_order_reds() { 
            long retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetNumStabLimOrderReds, (cvode_mem, &retval));
            return retval;
        }

        double get_tol_scale_factor() { 
            double retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetTolScaleFactor, (cvode_mem, &retval));
            return retval;
        }

        void get_err_weights(np_array<double> &eweight) {
            array_nvector nv(eweight);
            CHECK_SUNDIALS_RET(CVodeGetErrWeights, (cvode_mem, nv.ptr()));
        }

        void get_est_local_errors(np_array<double> &ele) {
            array_nvector nv(ele);
            CHECK_SUNDIALS_RET(CVodeGetEstLocalErrors, (cvode_mem, nv.ptr()));
        }

        bp::tuple get_integrator_stats() {
            long nsteps = 0, nfevals = 0, nlinsetups =  0, netfails = 0;
            int qlast = 0, qcur = 0;
            double hinused = 0, hlast = 0, hcur = 0, tcur = 0;
            CHECK_SUNDIALS_RET(CVodeGetIntegratorStats, (cvode_mem, &nsteps, &nfevals, &nlinsetups, &netfails, &qlast, &qcur, &hinused, &hlast, &hcur, &tcur));
            // TODO: return a hash or object (?)
            return bp::make_tuple(nsteps, nfevals, nlinsetups, netfails, qlast, qcur, hinused, hlast, hcur, tcur);
        }

        long get_num_nonlin_solv_iters() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetNumNonlinSolvIters, (cvode_mem, &retval));
            return retval;
        }

        long get_num_nonlin_solv_conv_fails() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVodeGetNumNonlinSolvConvFails, (cvode_mem, &retval));
            return retval;
        }

        bp::tuple get_nonlin_solv_stats() {
            long nniters = 0, nncfails = 0;
            CHECK_SUNDIALS_RET(CVodeGetNonlinSolvStats, (cvode_mem, &nniters, &nncfails));
            // TODO: return a hash or object (?)
            return bp::make_tuple(nniters, nncfails);
        }

        static std::string get_return_flag_name(int flag);

        // direct linear solver optional output functions

        bp::tuple get_dls_work_space() {
            long lenrwLS = 0, leniwLS= 0;
            CHECK_SUNDIALS_RET(CVDlsGetWorkSpace, (cvode_mem, &lenrwLS, &leniwLS));
            return bp::make_tuple(lenrwLS, leniwLS);
        }

        long get_dls_num_jac_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVDlsGetNumJacEvals, (cvode_mem, &retval));
            return retval;
        }

        long get_dls_num_rhs_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVDlsGetNumRhsEvals, (cvode_mem, &retval));
            return retval;
        }

        int get_dls_last_flag() {
            sundials_long_param_t retval = 0;
            CHECK_SUNDIALS_RET(CVDlsGetLastFlag, (cvode_mem, &retval));
            return retval;
        }

        static std::string get_dls_return_flag_name(int flag);

        // diagonal linear solver optional output functions

        bp::tuple get_diag_work_space() {
            long lenrwLS = 0, leniwLS= 0;
            CHECK_SUNDIALS_RET(CVDiagGetWorkSpace, (cvode_mem, &lenrwLS, &leniwLS));
            return bp::make_tuple(lenrwLS, leniwLS);
        }

        long get_diag_num_rhs_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVDiagGetNumRhsEvals, (cvode_mem, &retval));
            return retval;
        }

        int get_diag_last_flag() {
            sundials_long_param_t retval = 0;
            CHECK_SUNDIALS_RET(CVDiagGetLastFlag, (cvode_mem, &retval));
            return retval;
        }

        static std::string get_diag_return_flag_name(int flag);

        // iterative linear solver optional output functions

        bp::tuple get_spils_work_space() {
            long lenrwLS = 0, leniwLS= 0;
            CHECK_SUNDIALS_RET(CVSpilsGetWorkSpace, (cvode_mem, &lenrwLS, &leniwLS));
            return bp::make_tuple(lenrwLS, leniwLS);
        }

        long get_spils_num_lin_iters() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVSpilsGetNumLinIters, (cvode_mem, &retval));
            return retval;
        }

        long get_spils_num_conv_fails() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVSpilsGetNumConvFails, (cvode_mem, &retval));
            return retval;
        }

        long get_spils_num_prec_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVSpilsGetNumPrecEvals, (cvode_mem, &retval));
            return retval;
        }

        long get_spils_num_prec_solves() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVSpilsGetNumPrecSolves, (cvode_mem, &retval));
            return retval;
        }

        long get_spils_num_jtimes_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVSpilsGetNumJtimesEvals, (cvode_mem, &retval));
            return retval;
        }

        long get_spils_num_rhs_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVSpilsGetNumRhsEvals, (cvode_mem, &retval));
            return retval;
        }

        int get_spils_last_flag() {
            sundials_long_param_t retval = 0;
            CHECK_SUNDIALS_RET(CVSpilsGetLastFlag, (cvode_mem, &retval));
            return retval;
        }

        static std::string get_spils_return_flag_name(int flag);

        // reinitialisation functions

        void reinit(double t0, const np_array<double> &y0) {
            array_nvector y0_nv(y0);
            CHECK_SUNDIALS_RET(CVodeReInit, (cvode_mem, t0, y0_nv.ptr()));
        }

        // preconditioner functions

        void band_prec_init(int n, int mu, int ml) {
            CHECK_SUNDIALS_RET(CVBandPrecInit, (cvode_mem, n, mu, ml));
        }

        bp::tuple get_band_prec_work_space() {
            long lenrwBP = 0, leniwBP = 0;
            CHECK_SUNDIALS_RET(CVBandPrecGetWorkSpace, (cvode_mem, &lenrwBP, &leniwBP));
            return bp::make_tuple(lenrwBP, leniwBP);
        }

        long get_band_prec_num_rhs_evals() {
            long retval = 0;
            CHECK_SUNDIALS_RET(CVBandPrecGetNumRhsEvals, (cvode_mem, &retval));
            return retval;
        }

    private:
        // Callbacks
        // Error processing
        static void error_callback(int error_code, const char *module, const char *function, char *msg, void *eh_data);

        // ODE right-hand side
        static int rhs_callback(realtype t, N_Vector y, N_Vector ydot, void *user_data) {
            cvode *cv = (cvode*) user_data;

            // call back into Python code
            finmag::util::scoped_gil_ensure gil_ensure;
            bp::object y_arr = nvector_to_array_object(y);
            bp::object ydot_arr = nvector_to_array_object(ydot);

            // TODO: catch exceptions here
            // The time iteration function (CVode) should not allocate memory, so an exception
            // propagating through sundials code should not result in a memory leak. Still this might be risky
            bp::object res_obj = cv->rhs_fn(t, y_arr, ydot_arr);
            bp::extract<int> res(res_obj);
            if (res.check()) {
                return res();
            } else {
                // Callback did not return an integer
                error_handler::set_error("Error in r.h.s. callback: User-supplied Python right-hand-side function must return an integer to indicate outcome of operation");
                return -1;
            }
        }

        // Jacobian information (direct method with dense Jacobian)
        static int dls_dense_jac_callback(sundials_long_param_t n, realtype t, N_Vector y, N_Vector fy, DlsMat Jac,
                    void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
//            cvode *cv = (cvode*) user_data;

            // call back into Python code
//            finmag::util::scoped_gil_ensure gil_ensure;
//            bp::object y_arr = nvector_to_array_object(y);
//            bp::object fy_arr = nvector_to_array_object(fy);
            ASSERT(false && "dls_dense_jac_callback not implemented");
            abort();
        }

        // Jacobian information (direct method with banded Jacobian)
        static int dls_band_jac_callback(sundials_long_param_t n, sundials_long_param_t mupper, sundials_long_param_t mlower, realtype t, N_Vector y, N_Vector fy,
                    DlsMat Jac, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
            ASSERT(false && "dls_band_jac_callback not implemented");
            abort();
        }

        // Jacobian information (matrix-vector product)
        static int spils_jac_times_vec_callback(N_Vector v, N_Vector Jv, realtype t, N_Vector y,
                    N_Vector fy, void *user_data, N_Vector tmp) {
            cvode *cv = (cvode*) user_data;

            // call back into Python code
            finmag::util::scoped_gil_ensure gil_ensure;
            bp::object v_arr = nvector_to_array_object(v);
            bp::object Jv_arr = nvector_to_array_object(Jv);
            bp::object y_arr = nvector_to_array_object(y);
            bp::object fy_arr = nvector_to_array_object(fy);
            bp::object tmp_arr = nvector_to_array_object(tmp);

            // TODO: catch exceptions here - see comment in rhs_callback
            bp::object res_obj = cv->spils_jac_times_vec_fn(v_arr, Jv_arr, t, y_arr, fy_arr, tmp_arr);

            bp::extract<int> res(res_obj);
            if (res.check()) {
                return res();
            } else {
                // Callback did not return an integer
                error_handler::set_error("Error in Jacobean-times-vector callback: User-supplied Python Jacobean-times-vector function must return an integer to indicate outcome of operation");
                return -1;
            }
        }

        // Preconditioning (linear system solution)
        static int spils_prec_solve_callback(realtype t, N_Vector y, N_Vector fy, N_Vector r, N_Vector z,
                    realtype gamma, realtype delta, int lr, void *user_data, N_Vector tmp) {
            cvode *cv = (cvode*) user_data;

            // call back into Python code
            finmag::util::scoped_gil_ensure gil_ensure;
            bp::object y_arr = nvector_to_array_object(y);
            bp::object fy_arr = nvector_to_array_object(fy);
            bp::object r_arr = nvector_to_array_object(r);
            bp::object z_arr = nvector_to_array_object(z);
            bp::object tmp_arr = nvector_to_array_object(tmp);

            // TODO: catch exceptions here - see comment in rhs_callback
            return bp::call<int>(cv->spils_prec_solve_fn.ptr(), t, y_arr, fy_arr, r_arr, z_arr, gamma, delta, lr, tmp_arr);
        }

        // Preconditioning (Jacobian data)
        static int spils_prec_setup_callback(realtype t, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr,
                    realtype gamma, void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
            cvode *cv = (cvode*) user_data;

            // call back into Python code
            finmag::util::scoped_gil_ensure gil_ensure;
            bp::object y_arr = nvector_to_array_object(y);
            bp::object fy_arr = nvector_to_array_object(fy);
            bp::object tmp1_arr = nvector_to_array_object(tmp1);
            bp::object tmp2_arr = nvector_to_array_object(tmp2);
            bp::object tmp3_arr = nvector_to_array_object(tmp3);

            // TODO: catch exceptions here - see comment in rhs_callback
            bp::object res = bp::call<bp::tuple>(cv->spils_prec_setup_fn.ptr(), t, y_arr, fy_arr, jok, gamma, tmp1_arr, tmp2_arr, tmp3_arr);
            int flag = bp::extract<int>(res[0]);
            *jcurPtr = bp::extract<bool>(res[1]);
            return flag;
        }

        void* cvode_mem;

        bp::object rhs_fn, dls_jac_fn, dls_band_jac_fn, spils_prec_setup_fn, spils_prec_solve_fn, spils_jac_times_vec_fn;
    };

    void register_sundials_cvode();

    // Error message handler function
    void cvode::error_callback(int error_code, const char *module, const char *function, char *msg, void *eh_data) {
        char buf[1024];
        buf[1023] = 0;

        std::string error_code_str;
        if (strcmp(module, "CVODE") == 0) {
            error_code_str = cvode::get_return_flag_name(error_code);
        } else if (strcmp(module, "CVSPGMR") == 0) {
            error_code_str = cvode::get_spils_return_flag_name(error_code);
        } else {
            error_code_str = boost::lexical_cast<std::string>(error_code);
        }

        snprintf(buf, 1023, "Error in %s:%s (%s): %s", module, function, error_code_str.c_str(), msg);

        error_handler::set_error(buf);
    }

    cvode::cvode(int lmm, int iter): cvode_mem(0) {
        if (lmm != CV_ADAMS && lmm != CV_BDF)
            throw std::invalid_argument("sundials_cvode: lmm parameter must be either CV_ADAMS or CV_BDF");
        if (iter != CV_NEWTON && iter != CV_FUNCTIONAL)
            throw std::invalid_argument("sundials_cvode: iter parameter must be either CV_NEWTON or CV_FUNCTIONAL");
        cvode_mem = CVodeCreate(lmm, iter);
        if (!cvode_mem) throw std::runtime_error("CVodeCreate returned NULL");

        // Fix bug in sundials: CVodeCreate does not set all of the vector fields to NULL
        // So when CVodeFree is called without a previous CVodeInit, a segfault occurs. Fail...
        CVodeMem cm = (CVodeMem) cvode_mem;
        cm->cv_Vabstol = 0;
        for (int i = 0; i < L_MAX; i++) cm->cv_zn[i] = 0;
        cm->cv_ewt = 0;
        cm->cv_y = 0;
        cm->cv_acor = 0;
        cm->cv_tempv = 0;
        cm->cv_ftemp = 0;

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

    void error_handler::set_error(const char *msg) {
        std::unique_ptr<std::string> old_msg(cvode_error.release());
        if (!old_msg.get()) {
            // set new error message
            cvode_error.reset(new std::string(msg));
        } else {
            // Append to existing error message
            std::unique_ptr<std::string> new_msg(new std::string(*old_msg + "\n" + msg));
            cvode_error.reset(new_msg.release());
        }
    }

    void cvode::init(const bp::object &f, double t0, const np_array<double>& y0) {
        rhs_fn = f;
        array_nvector y0_nvec(y0);
        CHECK_SUNDIALS_RET(CVodeInit, (cvode_mem, rhs_callback, t0, y0_nvec.ptr()));
        // TODO: add a flag that cvode has been initialised; raise exceptions if flag unset
    }

    void cvode::set_scalar_tolerances(double reltol, double abstol) {
        CHECK_SUNDIALS_RET(CVodeSStolerances, (cvode_mem, reltol, abstol));
    }

    double cvode::advance_time(double tout, const np_array<double> &yout, int itask) {
        array_nvector yout_nvec(yout);
        double tret = 0;
        // Release GIL while we are performing time integration
        finmag::util::scoped_gil_release gil_release;
        CHECK_SUNDIALS_RET(CVode, (cvode_mem, tout, yout_nvec.ptr(), &tout, itask));
        return tret;
    }

    boost::thread_specific_ptr<std::string> error_handler::cvode_error;

    std::string cvode::get_return_flag_name(int flag) {
        switch (flag) {
        case CV_SUCCESS: return "CV_SUCCESS";
        case CV_TSTOP_RETURN: return "CV_TSTOP_RETURN";
        case CV_ROOT_RETURN: return "CV_ROOT_RETURN";
        case CV_WARNING: return "CV_WARNING";
        case CV_TOO_MUCH_WORK: return "CV_TOO_MUCH_WORK";
        case CV_TOO_MUCH_ACC: return "CV_TOO_MUCH_ACC";
        case CV_ERR_FAILURE: return "CV_ERR_FAILURE";
        case CV_CONV_FAILURE: return "CV_CONV_FAILURE";
        case CV_LINIT_FAIL: return "CV_LINIT_FAIL";
        case CV_LSETUP_FAIL: return "CV_LSETUP_FAIL";
        case CV_LSOLVE_FAIL: return "CV_LSOLVE_FAIL";
        case CV_RHSFUNC_FAIL: return "CV_RHSFUNC_FAIL";
        case CV_FIRST_RHSFUNC_ERR: return "CV_FIRST_RHSFUNC_ERR";
        case CV_REPTD_RHSFUNC_ERR: return "CV_REPTD_RHSFUNC_ERR";
        case CV_UNREC_RHSFUNC_ERR: return "CV_UNREC_RHSFUNC_ERR";
        case CV_RTFUNC_FAIL: return "CV_RTFUNC_FAIL";
        case CV_MEM_FAIL: return "CV_MEM_FAIL";
        case CV_MEM_NULL: return "CV_MEM_NULL";
        case CV_ILL_INPUT: return "CV_ILL_INPUT";
        case CV_NO_MALLOC: return "CV_NO_MALLOC";
        case CV_BAD_K: return "CV_BAD_K";
        case CV_BAD_T: return "CV_BAD_T";
        case CV_BAD_DKY: return "CV_BAD_DKY";
        case CV_TOO_CLOSE: return "CV_TOO_CLOSE";
        default: return boost::lexical_cast<std::string>(flag);
        }
    }

    std::string cvode::get_spils_return_flag_name(int flag) {
        switch (flag) {
        case CVSPILS_SUCCESS: return "CVSPILS_SUCCESS";
        case CVSPILS_MEM_NULL: return "CVSPILS_MEM_NULL";
        case CVSPILS_LMEM_NULL: return "CVSPILS_LMEM_NULL";
        case CVSPILS_ILL_INPUT: return "CVSPILS_ILL_INPUT";
        case CVSPILS_MEM_FAIL: return "CVSPILS_MEM_FAIL";
        case CVSPILS_PMEM_NULL: return "CVSPILS_PMEM_NULL";
        default: return boost::lexical_cast<std::string>(flag);
        }
    }


    void register_sundials_cvode() {
        using namespace bp;

        def("get_sundials_version", &get_sundials_version);

        class_<cvode> cv("cvode", init<int, int>(args("lmm", "iter")));

        // initialisation functions
        cv.def("init", &cvode::init, (arg("f"), arg("t0"), arg("y0")));
        cv.def("set_scalar_tolerances", &cvode::set_scalar_tolerances, (arg("reltol"), arg("abstol")));
        // linear soiver specification functions
        cv.def("set_linear_solver_dense", &cvode::set_linear_solver_dense, (arg("n")));
        cv.def("set_linear_solver_lapack_dense", &cvode::set_linear_solver_lapack_dense, (arg("n")));
        cv.def("set_linear_solver_band", &cvode::set_linear_solver_band, (arg("n"), arg("mupper"), arg("mlower")));
        cv.def("set_linear_solver_lapack_band", &cvode::set_linear_solver_lapack_band, (arg("n"), arg("mupper"), arg("mlower")));
        cv.def("set_linear_solver_diag", &cvode::set_linear_solver_diag);
        cv.def("set_linear_solver_sp_gmr", &cvode::set_linear_solver_sp_gmr, (arg("pretype"), arg("maxl")=0));
        cv.def("set_linear_solver_sp_bcg", &cvode::set_linear_solver_sp_bcg, (arg("pretype"), arg("maxl")=0));
        cv.def("set_linear_solver_sp_tfqmr", &cvode::set_linear_solver_sp_tfqmr, (arg("pretype"), arg("maxl")=0));
        // solver functions
        cv.def("advance_time", &cvode::advance_time, (arg("tout"), arg("yout"), arg("itask")=CV_NORMAL));
        // main solver optional input functions
        cv.def("set_max_ord", &cvode::set_max_ord, (arg("max_order")));
        cv.def("set_max_num_steps", &cvode::set_max_num_steps, (arg("max_steps")));
        cv.def("set_max_hnil_warns", &cvode::set_max_hnil_warns, (arg("max_hnil")));
        cv.def("set_stab_lim_det", &cvode::set_stab_lim_det, (arg("stldet")));
        cv.def("set_init_step", &cvode::set_init_step, (arg("initial_step_size")));
        cv.def("set_min_step_size", &cvode::set_min_step_size, (arg("min_step_size")));
        cv.def("set_max_step_size", &cvode::set_max_step_size, (arg("max_step_size")));
        cv.def("set_stop_time", &cvode::set_stop_time, (arg("stop_time")));
        cv.def("set_max_err_test_fails", &cvode::set_max_err_test_fails, (arg("max_err_test_fails")));
        cv.def("set_max_nonlin_iters", &cvode::set_max_nonlin_iters, (arg("max_nonlin_iters")));
        cv.def("set_max_conv_fails", &cvode::set_max_conv_fails, (arg("max_conv_fails")));
        cv.def("set_nonlin_conv_coef", &cvode::set_nonlin_conv_coef, (arg("nonlin_conv_coef")));
        cv.def("set_iter_type", &cvode::set_iter_type, (arg("iter_type")));
        // direct linear solver optional input functions
        cv.def("set_dls_jac_fn", &cvode::set_dls_jac_fn);
        cv.def("set_dls_band_jac_fn", &cvode::set_dls_band_jac_fn);
        // iterative linear solver optional input functions
        cv.def("set_spils_preconditioner", &cvode::set_spils_preconditioner, (arg("psetup"), arg("psolve")));
        cv.def("set_spils_jac_times_vec_fn", &cvode::set_spils_jac_times_vec_fn, (arg("jtimes")));
        cv.def("set_spils_prec_type", &cvode::set_spils_prec_type, (arg("pretype")));
        cv.def("set_spils_gs_type", &cvode::set_spils_gs_type, (arg("gstype")));
        cv.def("set_spils_eps_lin", &cvode::set_spils_eps_lin, (arg("eplifac")));
        cv.def("set_spils_maxl", &cvode::set_spils_maxl, (arg("maxl")));
        // optional output functions
        cv.def("get_work_space", &cvode::get_work_space);
        cv.def("get_num_steps", &cvode::get_num_steps);
        cv.def("get_num_rhs_evals", &cvode::get_num_rhs_evals);
        cv.def("get_num_lin_solv_setups", &cvode::get_num_lin_solv_setups);
        cv.def("get_num_err_test_fails", &cvode::get_num_err_test_fails);
        cv.def("get_last_order", &cvode::get_last_order);
        cv.def("get_current_order", &cvode::get_current_order);
        cv.def("get_last_step", &cvode::get_last_step);
        cv.def("get_current_step", &cvode::get_current_step);
        cv.def("get_actual_init_step", &cvode::get_actual_init_step);
        cv.def("get_current_time", &cvode::get_current_time);
        cv.def("get_num_stab_lim_order_reds", &cvode::get_num_stab_lim_order_reds);
        cv.def("get_tol_scale_factor", &cvode::get_tol_scale_factor);
        cv.def("get_err_weights", &cvode::get_err_weights, (arg("eweights")));
        cv.def("get_est_local_errors", &cvode::get_est_local_errors, (arg("ele")));
        cv.def("get_integrator_stats", &cvode::get_integrator_stats);
        cv.def("get_num_nonlin_solv_iters", &cvode::get_num_nonlin_solv_iters);
        cv.def("get_num_nonlin_solv_conv_fails", &cvode::get_num_nonlin_solv_conv_fails);
        cv.def("get_nonlin_solv_stats", &cvode::get_nonlin_solv_stats);
        // direct linear solver optional output functions
        cv.def("get_dls_work_space", &cvode::get_dls_work_space);
        cv.def("get_dls_num_jac_evals", &cvode::get_dls_num_jac_evals);
        cv.def("get_dls_num_rhs_evals", &cvode::get_dls_num_rhs_evals);
        cv.def("get_dls_last_flag", &cvode::get_dls_last_flag);
        // diagonal linear solver optional output functions
        cv.def("get_diag_work_space", &cvode::get_diag_work_space);
        cv.def("get_diag_num_rhs_evals", &cvode::get_diag_num_rhs_evals);
        cv.def("get_diag_last_flag", &cvode::get_diag_last_flag);
        // iterative linear solver optional output functions
        cv.def("get_spils_work_space", &cvode::get_spils_work_space);
        cv.def("get_spils_num_lin_iters", &cvode::get_spils_num_lin_iters);
        cv.def("get_spils_num_conv_fails", &cvode::get_spils_num_conv_fails);
        cv.def("get_spils_num_prec_evals", &cvode::get_spils_num_prec_evals);
        cv.def("get_spils_num_prec_solves", &cvode::get_spils_num_prec_solves);
        cv.def("get_spils_num_jtimes_evals", &cvode::get_spils_num_jtimes_evals);
        cv.def("get_spils_num_rhs_evals", &cvode::get_spils_num_rhs_evals);
        cv.def("get_spils_last_flag", &cvode::get_spils_last_flag);
        // reinitialisation functions
        cv.def("reinit", &cvode::reinit, (arg("t0"), arg("y0")));
        // preconditioner functions
        cv.def("band_prec_init", &cvode::band_prec_init, (arg("n"), arg("mu"), arg("ml")));
        cv.def("get_band_prec_work_space", &cvode::get_band_prec_work_space);
        cv.def("get_band_prec_num_rhs_evals", &cvode::get_band_prec_num_rhs_evals);

        scope().attr("CV_ADAMS") = int(CV_ADAMS);
        scope().attr("CV_BDF") = int(CV_BDF);

        scope().attr("CV_FUNCTIONAL") = int(CV_FUNCTIONAL);
        scope().attr("CV_NEWTON") = int(CV_NEWTON);

        scope().attr("CV_NORMAL") = int(CV_NORMAL);
        scope().attr("CV_ONE_STEP") = int(CV_ONE_STEP);

        scope().attr("PREC_NONE") = int(PREC_NONE);
        scope().attr("PREC_LEFT") = int(PREC_LEFT);
        scope().attr("PREC_RIGHT") = int(PREC_RIGHT);
        scope().attr("PREC_BOTH") = int(PREC_BOTH);

        scope().attr("MODIFIED_GS") = int(MODIFIED_GS);
        scope().attr("CLASSICAL_GS") = int(CLASSICAL_GS);
    }
}}

#endif
