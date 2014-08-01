#include <memory>
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

        private:
            GenericVector const& magnetisation;
            GenericVector const& effective_field;
            GenericVector& derivative;
            std::shared_ptr<GenericVector> alpha;
            double gamma;
    };
}}
