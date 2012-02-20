#include "finmag_includes.h"

#include "numpy_malloc.h"

namespace finmag { namespace util {
    namespace {
        extern "C" void * numpy_malloc(size_t len, size_t el_size) {
        }

        extern "C" void numpy_free(void *ptr) {
        }
    }

    void register_numpy_malloc() {
        set_nvector_custom_data_malloc(numpy_malloc, numpy_free);
    }
}}