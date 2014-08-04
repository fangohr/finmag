#include <iostream>
#include <string>
#include <stdexcept>
#include "equation.h"
#include "terms.h"
#include "derivatives.h"

namespace dolfin { namespace finmag {

    void check_size(const GenericVector& v, const GenericVector& w, const std::string& msg) {
        if (v.size() != w.size()) {
            throw std::length_error(msg + ": size " + std::to_string(v.size()) + " vs. " + std::to_string(w.size()));
        }
    }

    Equation::Equation(GenericVector const& m,
                       GenericVector const& H,
                       GenericVector& dmdt) :
        magnetisation(m),
        effective_field(H),
        derivative(dmdt)
    {
        check_size(magnetisation, effective_field, "m and H");
        check_size(magnetisation, derivative, "m and dmdt");

        std::shared_ptr<GenericVector> alpha(nullptr);
        double gamma {2.210173e5};
        double parallel_relaxation_rate {1e-12};
        bool do_precession {true};
    }

    std::shared_ptr<GenericVector> Equation::get_alpha() const { return alpha; } 
    void Equation::set_alpha(std::shared_ptr<GenericVector> const& value) { alpha = value; }
    double Equation::get_gamma() const { return gamma; }
    void Equation::set_gamma(double value) { gamma = value; }
    double Equation::get_parallel_relaxation_rate() const { return parallel_relaxation_rate; }
    void Equation::set_parallel_relaxation_rate(double value) { parallel_relaxation_rate = value; }
    bool Equation::get_do_precession() const { return do_precession; }
    void Equation::set_do_precession(bool value) { do_precession = value; }

    /* Solve the equation for dm/dt, writing the solution into the vector
     * that was passed during initialisation of the class. */
    void Equation::solve() {
        if (!alpha) throw std::runtime_error("alpha was not set");

        std::vector<double> a, m, H, dmdt;
        alpha->get_local(a);
        magnetisation.get_local(m);
        effective_field.get_local(H);
        derivative.get_local(dmdt);

        std::vector<double>::size_type x=0, y=0, z=0;
        /* When we iterate over the nodes the iteration counter can be used to
         * access the value of scalar fields, since there is exactly one degree
         * of freedom per node. Vector fields have 3 degrees of freedom per node
         * and we thus multiply the iteration counter by 3 to get the first
         * component of the vector fields (which is the x-component). */
        for (std::vector<double>::size_type node=0; node < a.size(); ++node) {
            x = 3 * node; y = x + 1; z = x + 2;
            damping(a[node], gamma, m[x], m[y], m[z], H[x], H[y], H[z], dmdt[x], dmdt[y], dmdt[z]);
            if (do_precession) {
                precession(a[node], gamma, m[x], m[y], m[z], H[x], H[y], H[z], dmdt[x], dmdt[y], dmdt[z]);
            }
            relaxation(parallel_relaxation_rate, m[x], m[y], m[z], dmdt[x], dmdt[y], dmdt[z]);
        }

        derivative.set_local(dmdt);
        derivative.apply("");
    }
}}