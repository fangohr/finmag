#include <iostream>
#include "equation.h"
#include "terms.h"

namespace dolfin { namespace finmag {
    void print_info(const GenericVector& v) {
        std::cout << "size of v " << v.size() << std::endl;
    }

    Equation::Equation(const GenericVector& magnetisation,
                       const GenericVector& effective_field,
                       GenericVector& derivative) :
        m(magnetisation),
        H(effective_field),
        dmdt(derivative)
    {}

    /* Solve the equation for dm/dt, writing the solution into the vector
     * that was passed during initialisation of the class. */
    void Equation::solve() {
        std::cout << "Soooolving." << std::endl;
    }
}}
