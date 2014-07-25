#ifndef __FOO_H
#define __FOO_H

#include <dolfin.h>
//#include <dolfin/function/Function.h>


namespace dolfin
{
  class Foo
  {
  public:
    Foo();
    void bar(const Function& u);
    void bar2(Vector& a, Vector& b, double c, double d);
  };
}
#endif
