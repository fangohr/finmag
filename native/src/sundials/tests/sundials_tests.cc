#include "finmag_includes.h"

#include <boost/test/unit_test.hpp>

#include "sundials/numpy_malloc.h"

struct init_sundials {
    init_sundials() { finmag::sundials::register_numpy_malloc(); }
};

// Only perform the initialisation once
struct init_fixture {
    init_fixture() { static init_sundials init; }
};

BOOST_FIXTURE_TEST_SUITE(sundials_tests, init_fixture)

const int VALGRIND_LEAK_THRESHOLD = 10000000;
const int N = VALGRIND_LEAK_THRESHOLD/sizeof(double) + 1;

BOOST_AUTO_TEST_CASE(test_numpy_malloc)
{
    double *arr = (double*) finmag::sundials::numpy_malloc(N, sizeof(double));
    for (int i = 0; i < N; i+= N/1000) arr[i] = 17;
    finmag::sundials::numpy_free(arr);
}

BOOST_AUTO_TEST_CASE(test_array_to_nvector)
{
    np_array<double> arr(N);
    finmag::sundials::array_nvector nvec(arr);
    // assign to numpy array
    arr.data()[0] = 3.14;
    // check NVector data
    BOOST_CHECK_EQUAL(NV_Ith_S(nvec.ptr(), 0), 3.14);
}

BOOST_AUTO_TEST_CASE(test_nvector_to_array)
{
    N_Vector vec = N_VNew_Serial(N);
    np_array<double> arr = finmag::sundials::nvector_to_array(vec);
    // assign to numpy array
    arr.data()[0] = 3.14;
    // check NVector data
    BOOST_CHECK_EQUAL(NV_Ith_S(vec, 0), 3.14);
    // free the original NVector
    N_VDestroy_Serial(vec);
    // array data still usable
    for (int i = 0; i < N; i+=   N/1000) arr.data()[i] = 17;
}

BOOST_AUTO_TEST_CASE(test_nvector_to_array_object)
{
    N_Vector vec = N_VNew_Serial(N);
    bp::object obj = finmag::sundials::nvector_to_array_object(vec);
    np_array<double> arr = bp::extract<np_array<double> >(obj);
    // assign to numpy array
    arr.data()[0] = 3.14;
    // check NVector data
    BOOST_CHECK_EQUAL(NV_Ith_S(vec, 0), 3.14);
    // free the original NVector
    N_VDestroy_Serial(vec);
    // array data still usable
    for (int i = 0; i < N; i+=   N/1000) arr.data()[i] = 17;
}

BOOST_AUTO_TEST_SUITE_END()
