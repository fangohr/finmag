#include <memory>
#include <vector>
#include <dolfin/la/GenericVector.h>
#include "terms.h"

/* compile_extension_module needs code to be wrapped in the dolfin namespace */
namespace dolfin { namespace finmag {
    class Equation {
        public:
            Equation(GenericVector const& m,
                     GenericVector const& H,
                     GenericVector& dmdt);

            void solve();
           
            std::shared_ptr<GenericVector> get_pinned_nodes() const;
            void set_pinned_nodes(std::shared_ptr<GenericVector> const& value);
            std::shared_ptr<GenericVector> get_saturation_magnetisation() const;
            void set_saturation_magnetisation(std::shared_ptr<GenericVector> const& value);
            std::shared_ptr<GenericVector> get_current_density() const;
            void set_current_density(std::shared_ptr<GenericVector> const& value);
            std::shared_ptr<GenericVector> get_alpha() const;
            void set_alpha(std::shared_ptr<GenericVector> const& value);
            double get_gamma() const;
            void set_gamma(double value);
            double get_parallel_relaxation_rate() const;
            void set_parallel_relaxation_rate(double value);
            bool get_do_precession() const;
            void set_do_precession(bool value);

            void slonczewski(double d, double P, Array<double> const& p, double lambda, double epsilonprime);
            void slonczewski_disable();
            bool slonczewski_status() const;

        private:
            GenericVector const& magnetisation;
            GenericVector const& effective_field;
            GenericVector& derivative;
            std::shared_ptr<GenericVector> pinned_nodes;
            std::shared_ptr<GenericVector> saturation_magnetisation;
            std::shared_ptr<GenericVector> current_density;
            std::shared_ptr<GenericVector> alpha;
            double gamma;
            double parallel_relaxation_rate;
            bool do_precession;
            bool do_slonczewski;
            std::unique_ptr<Slonczewski> sl_stt;
    };
}}
