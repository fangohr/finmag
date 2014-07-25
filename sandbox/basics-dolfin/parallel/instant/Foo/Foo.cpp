#include "Foo.h"

using namespace dolfin;

Foo::Foo()
{

}

void Foo::bar(const Function& u)
{

}

void Foo::bar2(Vector& a, Vector& b, double c, double d) {
  for (unsigned int i=0; i < a.size(); i++) {
    b.setitem(i, d*a[i] + c); 
  }
}
