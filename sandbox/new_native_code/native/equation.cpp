#include <iostream>
#include <string>
#include <stdexcept>
#include "equation.h"
#include "terms.h"

void check_size(const dolfin::GenericVector& v, const dolfin::GenericVector& w, const std::string& msg) {
    if (v.size() != w.size()) {
        throw std::length_error(msg + ": size " + std::to_string(v.size()) + " vs. " + std::to_string(w.size()));
    }
} 

namespace dolfin { namespace finmag {
    Equation::Equation(GenericVector const& m,
                       GenericVector const& H,
                       GenericVector& dmdt) :
        magnetisation(m),
        effective_field(H),
        derivative(dmdt)
    {
        check_size(magnetisation, effective_field, "m and H");
        check_size(magnetisation, derivative, "m and dmdt");
    }

    /* Solve the equation for dm/dt, writing the solution into the vector
     * that was passed during initialisation of the class. */
    void Equation::solve() {
        std::vector<double> m, H, dmdt;
        magnetisation.get_local(m);
        effective_field.get_local(H);
        derivative.get_local(dmdt);

        for (std::vector<double>::size_type i=0; i < m.size(); i += 3) {
            damping(1, 1, m[i], m[i+1], m[i+2], H[i], H[i+1], H[i+2], dmdt[i], dmdt[i+1], dmdt[i+2]);
        }

        derivative.set_local(dmdt);
        derivative.apply("");
    }
}}
