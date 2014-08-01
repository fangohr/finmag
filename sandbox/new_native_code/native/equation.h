#include <dolfin/la/GenericVector.h>

/* compile_extension_module needs code to be wrapped in the dolfin namespace */
namespace dolfin { namespace finmag {
    class Equation {
        public:
            Equation(GenericVector const& m,
                     GenericVector const& H,
                     GenericVector& dmdt);
            void solve();

        private:
            GenericVector const& magnetisation;
            GenericVector const& effective_field;
            GenericVector& derivative;
    };
}}
