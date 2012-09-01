#include "finmag_includes.h"
#include "util/np_array.h"

namespace finmag { namespace llg {

    namespace {
        void helloWorld() {
            printf("This is the StochasticHeunIntegrator.\n");
        }

        class StochasticHeunIntegrator {
        public:
            void helloWorld() {
                printf("This is the StochasticHeunIntegrator.\n");
            }
        };

    }

    void register_heun() {
        using namespace boost::python;

        def("helloWorld", &helloWorld);

        class_<StochasticHeunIntegrator>("StochasticHeunIntegrator", init<>())
            .def("helloWorld", &StochasticHeunIntegrator::helloWorld);
    }

}}
