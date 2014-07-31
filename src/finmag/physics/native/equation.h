#include <dolfin/la/GenericVector.h>

/* compile_extension_module needs code to be wrapped in the dolfin namespace */
namespace dolfin { namespace finmag {
    class Equation {
        public:
            Equation(const GenericVector& magnetisation,
                     const GenericVector& effective_field,
                     GenericVector& derivative);
            void solve();

        private:
            const GenericVector& m;
            const GenericVector& H;
            GenericVector& dmdt;
    };
}}
