// np_array.h assumes that the precompiled header file finmag_includes.h has been included previously
// here, to speed up compilation we provide a simplified finmag_includes.h with what is only necessary for the demo
#include "finmag_includes.h"
#include "../../../native/src/util/np_array.h"

// Computes the trace of a matrix
double trace(const np_array<double> &m)
{
    m.check_ndim(2, "trace: m");
    int n = m.dim()[0];
    m.check_shape(n, n, "trace: m");

    double sum = 0;
    for (int i = 0; i < n; i++) sum += m(i)[i];

    return sum;
}

BOOST_PYTHON_MODULE(demo3_module)
{
    initialise_np_array();

    bp::scope().attr("__doc__") = "Boost.Python Demo #3 - Numpy";

    bp::def("trace", &trace, bp::args("m"), "Computes the trace of a matrix");
}
