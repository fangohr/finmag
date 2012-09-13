#include "finmag_includes.h"
#include "util/np_array.h"

namespace finmag { namespace llg {

    namespace {
        class StochasticHeunIntegrator {
        public:
            StochasticHeunIntegrator(
                const np_array<double> &y,
                bp::object A_callback,
                bp::object B_callback,
                double dt
            ): y(y), A(A_callback), B(B_callback), dt(dt) {
            }
            ~StochasticHeunIntegrator() {}

            void run_until() {

            }
            
            void step() {
            }

            void helloWorld() {
                printf("This is the StochasticHeunIntegrator.\n");
            }
        private:
            const np_array<double> y;
            bp::object A, B;
            double const dt;
        };

    }

    void register_heun() {
        using namespace bp;

        class_<StochasticHeunIntegrator>("StochasticHeunIntegrator",
            init<np_array<double>, object, object, double>())
            .def("helloWorld", &StochasticHeunIntegrator::helloWorld)
            .def("step", &StochasticHeunIntegrator::step)
            .def("run_until", &StochasticHeunIntegrator::run_until);
    }

}}
