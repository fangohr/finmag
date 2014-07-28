#include <iostream>
#include "Foo.h"
#include "Bar.h"

using namespace dolfin;

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

    //for(size_t i=0; i < norm.size(); ++i) {
    // TODO: compute norm
    //}

    norm.set_local(local_norm);
}
