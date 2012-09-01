#include "finmag_includes.h"
#include "util/np_array.h"
#include "heun.h"

namespace finmag { namespace llg {

    namespace {
        void helloWorld() {
            printf("This is the StochasticHeunIntegrator.\n");
        }
    }

    void register_heun() {
        using namespace boost::python;
        def("helloWorld", &helloWorld);
    }

}}
