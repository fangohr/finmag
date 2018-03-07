#ifndef __FOO_H
#define __FOO_H

#include <dolfin/function/Function.h>
#include <dolfin/la/GenericVector.h>

namespace dolfin
{
  class Foo
  {
  public:
    Foo();
    void foo(const Function& u);
    void foo2(const GenericVector& v);
    void norm(const GenericVector& v, GenericVector& norm);
  };
}
#endif
