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
    Equation::Equation(const GenericVector& magnetisation,
                       const GenericVector& effective_field,
                       GenericVector& derivative) :
        m(magnetisation),
        H(effective_field),
        dmdt(derivative)
    {
        check_size(m, H, "m and H");
        check_size(m, dmdt, "m and dmdt");
    }

    /* Solve the equation for dm/dt, writing the solution into the vector
     * that was passed during initialisation of the class. */
    void Equation::solve() {
        std::cout << "Soooolving." << std::endl;
    }
}}
