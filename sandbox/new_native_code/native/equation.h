#include <memory>
#include <bitset>
#include <dolfin/la/GenericVector.h>

/* compile_extension_module needs code to be wrapped in the dolfin namespace */
namespace dolfin { namespace finmag {
    class Equation {
        public:
            Equation(GenericVector const& m,
                     GenericVector const& H,
                     GenericVector& dmdt);

            void solve();
           
            std::shared_ptr<GenericVector> get_alpha() const;
            void set_alpha(std::shared_ptr<GenericVector> const& value);
            double get_gamma() const;
            void set_gamma(double value);
            double get_parallel_relaxation_rate() const;
            void set_parallel_relaxation_rate(double value);
            bool get_do_precession() const;
            void set_do_precession(bool value);

        private:
            GenericVector const& magnetisation;
            GenericVector const& effective_field;
            GenericVector& derivative;
            std::shared_ptr<GenericVector> alpha;
            double gamma;
            double parallel_relaxation_rate;
            bool do_precession;
    };
}}
