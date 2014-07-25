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

void Foo::bar2(Vector& a, Vector& b, double c, double d) {
  for (unsigned int i=0; i < a.size(); i++) {
    b.setitem(i, d*a[i] + c); 
  }
}
