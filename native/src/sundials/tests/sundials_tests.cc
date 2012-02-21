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

BOOST_AUTO_TEST_CASE(test_numpy_malloc)
{
    double *arr = (double*) finmag::sundials::numpy_malloc(10, sizeof(double));
    for (int i = 0; i < 10; i++) arr[i] = 17;
    finmag::sundials::numpy_free(arr);
}

BOOST_AUTO_TEST_SUITE_END()
