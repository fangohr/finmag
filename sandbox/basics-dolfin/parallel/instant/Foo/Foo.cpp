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
