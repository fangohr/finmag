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

    class sundials_nvector {

    };

    class sundials_cvode {
    public:
        sundials_cvode(int lmm, int iter): cvode_mem(0) {
            if (lmm != CV_ADAMS && lmm != CV_BDF)
                throw std::invalid_argument("sundials_cvode: lmm parameter must be either CV_ADAMS or CV_BDF");
            if (iter != CV_NEWTON && iter != CV_FUNCTIONAL)
                throw std::invalid_argument("sundials_cvode: iter parameter must be either CV_NEWTON or CV_FUNCTIONAL");
            cvode_mem = CVodeCreate(lmm, iter);
            if (!cvode_mem) throw std::runtime_error("CVodeCreate returned NULL");
        }

        ~sundials_cvode() {
            if (cvode_mem) {
                CVodeFree(&cvode_mem);
                cvode_mem = 0;
            }
        }

        void init(bp::object rhs_func, double t0, const sundials_nvector& vec) {

        }

        void set_scalar_tolerances() {

        }

        void advance_time() {

        }
    private:
        void* cvode_mem;
    };

    void register_sundials_cvode() {
        using namespace bp;

        class_<sundials_cvode>("sundials_cvode", init<int, int>(args("lmm", "iter")))
        ;

        scope().attr("CV_ADAMS") = int(CV_ADAMS);
        scope().attr("CV_BDF") = int(CV_BDF);
        scope().attr("CV_NEWTON") = int(CV_NEWTON);
        scope().attr("CV_FUNCTIONAL") = int(CV_FUNCTIONAL);
    }
}}

#endif