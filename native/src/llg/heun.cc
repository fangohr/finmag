#include "finmag_includes.h"
#include "util/np_array.h"

namespace finmag { namespace llg {

    namespace {
        void helloWorld() {
            printf("This is the StochasticHeunIntegrator.\n");
        }

        class StochasticHeunIntegrator {
        public:
            StochasticHeunIntegrator() {}
            ~StochasticHeunIntegrator() {}
            
            void step() {
                // m_predicted = m + drift(m) * Dt + diffusion(m) * DW
                // m_corrected = m
                //      + 1/2 [drift(m_predicted) + drift(m)] * Dt 
                //      + 1/2 [diffusion(m_predicted) + diffusion(m)] * DW
            }

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
