#include <iostream>
#include "Foo.h"
#include "Bar.h"

using namespace dolfin;

static inline double square (double x) { return x*x; }

Foo::Foo()
{

}

void Foo::foo(const Function& u)
{
    std::cout << "Hello from Foo::foo." << std::endl;
    bar();
}

void Foo::foo2(const GenericVector& v)
{
    std::cout << "I was successfully passed a vector." << std::endl;
}

// write euclidean norm of `v` into `norm`
void Foo::norm(const GenericVector& v, GenericVector& norm)
{
    std::vector<double> local_v, local_norm;
    v.get_local(local_v);
    norm.get_local(local_norm);

    std::cout << "C++ debug" << std::endl;
    std::cout << "size of local v " << local_v.size() << std::endl;  // equivalent to v.local_size()
    std::cout << "size of local norm " << local_norm.size() << std::endl;

    // we're dealing with std::vectors now, not GenericVectors.
    // Alternatively, we could do the computation in one loop,
    // just showcasing std::transform 
    std::transform(local_v.begin(), local_v.end(), local_v.begin(), square); 
   
    for(std::vector<double>::size_type i=0; i < local_norm.size(); ++i) {
        local_norm[i] = local_v[i] + local_v[i+1] + local_v[i+2];
    }

    // again, this could have been done in the loop
    std::transform(local_norm.begin(), local_norm.end(), local_norm.begin(), (double(*)(double)) sqrt);

    norm.set_local(local_norm);
}
