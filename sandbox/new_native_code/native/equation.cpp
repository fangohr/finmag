#include <iostream>
#include <string>
#include <stdexcept>
#include "equation.h"
#include <dolfin/function/Function.h>

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
        derivative(dmdt),
        pinned_nodes(nullptr),
        saturation_magnetisation(nullptr),
        current_density(nullptr),
        alpha(nullptr),
        stt_slonczewski(nullptr),
        stt_zhangli(nullptr)
    {
        check_size(magnetisation, effective_field, "m and H");
        check_size(magnetisation, derivative, "m and dmdt");
        gamma = 2.210173e5;
        parallel_relaxation_rate = 1e-12;
        do_precession = true;
        do_slonczewski = false;
        do_zhangli = false;
    }

    std::shared_ptr<GenericVector> Equation::get_pinned_nodes() const { return pinned_nodes; } 
    void Equation::set_pinned_nodes(std::shared_ptr<GenericVector> const& value) { pinned_nodes = value; }
    std::shared_ptr<GenericVector> Equation::get_saturation_magnetisation() const { return saturation_magnetisation; } 
    void Equation::set_saturation_magnetisation(std::shared_ptr<GenericVector> const& value) { saturation_magnetisation = value; }
    std::shared_ptr<GenericVector> Equation::get_current_density() const { return current_density; } 
    void Equation::set_current_density(std::shared_ptr<GenericVector> const& value) { current_density = value; }
    std::shared_ptr<GenericVector> Equation::get_alpha() const { return alpha; } 
    void Equation::set_alpha(std::shared_ptr<GenericVector> const& value) { alpha = value; }
    double Equation::get_gamma() const { return gamma; }
    void Equation::set_gamma(double value) { gamma = value; }
    double Equation::get_parallel_relaxation_rate() const { return parallel_relaxation_rate; }
    void Equation::set_parallel_relaxation_rate(double value) { parallel_relaxation_rate = value; }
    bool Equation::get_do_precession() const { return do_precession; }
    void Equation::set_do_precession(bool value) { do_precession = value; }
    void Equation::slonczewski(double d, double P, Array<double> const& p, double lambda, double epsilonprime) {
        stt_slonczewski.reset(new Slonczewski(d, P, p, lambda, epsilonprime));
        do_slonczewski = true;
    }
    void Equation::slonczewski_disable() { do_slonczewski = false; }
    bool Equation::slonczewski_status() const { return do_slonczewski && current_density && saturation_magnetisation; }
    void Equation::zhangli(double u_0, double beta) {
        stt_zhangli.reset(new ZhangLi(u_0, beta));
        do_zhangli = true;
    }
    void Equation::zhangli_disable() { do_zhangli = false; }
    bool Equation::zhangli_status() const { return do_zhangli && current_density && saturation_magnetisation; }

    /* Solve the equation for dm/dt, writing the solution into the vector
     * that was passed during initialisation of the class. */
    void Equation::solve() {
        if (!alpha) throw std::runtime_error("alpha was not set");

        std::vector<double> a, m, H, dmdt, pinned, Ms, J;
        alpha->get_local(a);
        magnetisation.get_local(m);
        effective_field.get_local(H);
        derivative.get_local(dmdt);

        if (pinned_nodes) pinned_nodes->get_local(pinned);
        if (saturation_magnetisation) saturation_magnetisation->get_local(Ms);
        if (current_density) current_density->get_local(J);

        std::vector<double>::size_type x=0, y=0, z=0;
        /* When we iterate over the nodes the iteration counter can be used to
        * access the value of scalar fields, since there is exactly one degree
        * of freedom per node. Vector fields have 3 degrees of freedom per node
        * and we thus multiply the iteration counter by 3 to get the first
        * component of the vector fields (which is the x-component). */
        #pragma omp parallel for schedule(guided)
        for (std::vector<double>::size_type node=0; node < a.size(); ++node) {
            x = 3 * node; y = x + 1; z = x + 2;
            dmdt[x] = 0; dmdt[y] = 0; dmdt[z] = 0;

            if (pinned_nodes && pinned[node]) {
                continue; /* dmdt=0 on pinned nodes, so skip computation of it */
            }

            damping(a[node], gamma, m[x], m[y], m[z], H[x], H[y], H[z], dmdt[x], dmdt[y], dmdt[z]);
            if (do_precession) precession(a[node], gamma, m[x], m[y], m[z], H[x], H[y], H[z], dmdt[x], dmdt[y], dmdt[z]);
            relaxation(parallel_relaxation_rate, m[x], m[y], m[z], dmdt[x], dmdt[y], dmdt[z]);
            if (slonczewski_status()) stt_slonczewski->compute(a[node], gamma, J[node], Ms[node], m[x], m[y], m[z], dmdt[x], dmdt[y], dmdt[z]);
            if (zhangli_status()) stt_zhangli->compute(a[node], Ms[node], m[x], m[y], m[z], J[x], J[y], J[z], dmdt[x], dmdt[y], dmdt[z]);
        }

        derivative.set_local(dmdt);
        derivative.apply("");
    }
}}
