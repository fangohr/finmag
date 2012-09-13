#include "finmag_includes.h"
#include "util/np_array.h"
#include <vector>

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
                t = 0.0;
                dW = 0.0;
                y.check_ndim(1, "dimensions of y");
                length = y.dim()[0];
                std::vector<double> y_predicted (length, 0.0);
                std::vector<double> Ay (length, 0.0);
                std::vector<double> By (length, 0.0);
                std::vector<double> Ay_predicted (length, 0.0);
                std::vector<double> By_predicted (length, 0.0);
            }
            ~StochasticHeunIntegrator() {}

            void run_until(double new_t) {
                if (new_t <= t) {
                    return;
                }
                
                while (t < new_t) {
                    step();
                }
            }
            
            void step() {
                bp::call<void>(A.ptr(), y, t, Ay);
                bp::call<void>(B.ptr(), y, t, By);

                for (std::vector<double>::size_type i = 0; i != y_predicted.size(); i++) {
                    y_predicted[i] = *y[i] + Ay[i] * dt + By[i] * dW;
                }

                bp::call<void>(A.ptr(), y_predicted, t, Ay_predicted);
                bp::call<void>(B.ptr(), y_predicted, t, By_predicted);

                for (int i=0; i < length; i++) {
                    *y[i] += (Ay_predicted[i] + Ay[i]) * 0.5 * dt + (By_predicted[i] + By[i]) * 0.5 * dW;
                }

                t += dt;
            }

            void helloWorld() {
                printf("This is the StochasticHeunIntegrator.\n");
            }
        private:
            int length;
            const np_array<double> y;
            bp::object A, B;
            double const dt;
            double t, dW; // TODO: Gaussian random numbers for dW. 
            std::vector<double> y_predicted, Ay, By, Ay_predicted, By_predicted;
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
