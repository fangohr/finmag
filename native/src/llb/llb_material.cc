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

#include "util/np_array.h"

#include "llb_material.h"

namespace finmag { namespace llb {
    namespace {
        constexpr double PI = M_PI;
        constexpr double TC = 660.; // K
        constexpr double MU_0 = 4.*M_PI*1.E-7; //[T**2 m**3 / J]
        constexpr double A_FePt = 0.192625e-9; //[m]; lattice constant of Fept = 2 * A_FePt
        constexpr double MU_S_FePt = 2.995506845e-23; //[Am**2] magnetic moment
        // TODO: fix this, lattice is not cubic (0.371nm x 0.385nm)?
        constexpr double Ms_FePt = MU_S_FePt/(4*A_FePt*A_FePt*A_FePt); // A/m. 2 atoms per unit cell, lattice constant 2*A_FePt

        inline double ChiPar(double x){
            //Fit Parameter
            double a0 = 0.8;
            double a1 =-2.2e-07;
            double a2 = 1.95e-13;
            double a3 =-1.3e-17;
            double a4 =-4e-23;
            double a5 =-6.5076312364e-32;

            double chi_CGS = 0.0;
            double chi_SI  = 0.0;
            double chi = 0.0;

            if(x<TC) chi_CGS =(a0/660.*TC)/(4.*PI)/(TC-x)+a1*pow((TC-x),1.)+ a2*pow((TC-x),3.)+a3*pow((TC-x),4.)+ a4*pow((TC-x),6.)+ a5*pow((TC-x),9.);
            else chi_CGS = (1.1*1.4/660.*TC)/(4*PI)/(x-TC);
            chi_SI = 4*PI*chi_CGS;     // CHI_SI = 4*PI Chi_CGS
            chi = chi_SI*4*A_FePt*A_FePt*A_FePt/MU_S_FePt/MU_0; // extra term MS/MS_FePt (??)
            if(chi < 0.0) chi = fabs(chi); //[1/T]
            return(1./chi); // [T]
        }

        double ChiPer(double x){
          //Default for FePt
          double ChangeFunction = 1.0271;
          double Scale = 1.;

          //Fit Parameter
          double a0 = 0.00675;
          double a1 = 2.82269E-7;

          double chi_CGS = 0.0;
          double chi_SI  = 0.0;
          double chi     = 0.0;


          if(((ChangeFunction*TC)-x) > 1.E-3) chi_CGS = Scale*(a0+ a1*x);
          else chi_CGS = (0.9*1.4/660.*TC)/(4*PI)/(x-TC);
          chi_SI = 4*PI*chi_CGS;     // CHI_SI = 4*PI Chi_CGS
          chi = fabs(chi_SI*4*A_FePt*A_FePt*A_FePt/MU_S_FePt/MU_0);//[1/T]
          return(1./chi); // [T]
        }

        double Exchange(double x){
          //Fit Parameter
            double a0 =  3.90858143659231e-13;
            double a1 =  5.65571902911896e-11;
            double a2 = -1.11221431025254e-10;
            double a3 =  1.67761522644194e-10;
            double a4 = -1.38437771856782e-10;
            double a5 =  4.6483423884759e-11;

            double zz = 0.0;

            if((TC-x) > 1.E-3) zz = a0 + a1*(TC-x)/TC + a2*pow((TC-x)/TC,2.) + a3*pow((TC-x)/TC,3.)+a4*pow((TC-x)/TC,4.)+a5*pow((TC-x)/TC,5.); //[J/m]
            else zz = 0.0;
            return(zz); //[J] with cell size A
        }

        class LLBFePt {
        public:
            double T_C() const { return TC; }

            double M_s() const { return Ms_FePt; }

            double m_e(double x) const {
                if (x != 0.0) {
                    if (TC - x > 1.E-3) {
                        double a = (TC-x)/TC;
                        double a2 = a*a;
                        double a4 = a2*a2;
                        return sqrt(a)*1.3 - a*.12-a2*.51+a*a2*.37-a4*.01-a4*a4*.03;
                    }
                    else {
                        return 0.0;
                    }
                } else {
                    return 1.0;
                }
            }
            double inv_chi_par(double T) const { return ChiPar(T) * (1./MU_0); }

            double inv_chi_perp(double T) const { return ChiPer(T) * (1./MU_0); }
            double A(double T) const { return Exchange(T); }

            //Returns a 4 x n array: (m_e, A, inv_chi_perp, inv_chi_par) x n
            np_array<double> compute_parameters(const np_array<double> & T_arr) {
                T_arr.check_ndim(1, "compute_parameters: T");
                int n = T_arr.dim()[0];
                double *T = T_arr.data();

                np_array<double> res(4, n);

                double *m_e_val = res(0), *A_val = res(1), *inv_chi_perp_val = res(2), *inv_chi_par_val = res(3);

                #pragma omp parallel for schedule(guided)
                for (int i = 0; i < n; i++) {
                    double temp = T[i];
                    m_e_val[i] = m_e(temp);
                    A_val[i] = A(temp);
                    inv_chi_perp_val[i] = inv_chi_perp(temp);
                    inv_chi_par_val[i] = inv_chi_par(temp);
                }

                return res;
            }
        };

    }

    void register_llb_material() {
        using namespace bp;

        class_<LLBFePt>("LLBFePt", init<>())
            .def("T_C", &LLBFePt::T_C)
            .def("M_s", &LLBFePt::M_s)
            .def("A", &LLBFePt::A)
            .def("m_e", &LLBFePt::m_e)
            .def("inv_chi_par", &LLBFePt::inv_chi_par)
            .def("inv_chi_perp", &LLBFePt::inv_chi_perp)
            .def("compute_parameters", &LLBFePt::compute_parameters)
        ;
    }
}}