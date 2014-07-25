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

void Foo::norm(const GenericVector& v, const GenericVector& norm)
{
    std::vector<double> 

    for(size_t i=0; i < norm.size(); ++i) {
        norm[i]
        
    }
}
