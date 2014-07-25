#ifndef __FOO_H
#define __FOO_H

#include <dolfin/function/Function.h>

namespace dolfin
{
  class Foo
  {
  public:
    Foo();
    void foo(const Function& u);
  };
}
#endif
