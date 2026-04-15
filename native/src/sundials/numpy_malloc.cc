/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 */

#include "finmag_includes.h"

#include "numpy_malloc.h"
#include "util/python_threading.h"
namespace finmag { namespace sundials {
    namespace {
	#if NPY_FEATURE_VERSION < 7
	// numpy 1.6 and below
	inline void py_array_set_base(PyArrayObject *arr, PyObject *obj) {
	        PyArray_BASE(arr) = obj;
	}
	#else
	// numpy 1.7
	inline void py_array_set_base(PyArrayObject *arr, PyObject *obj) {
	        PyArray_SetBaseObject(arr, obj);
	}
	#endif


        static const unsigned long DATA_MAGIC = 0xF6833C73196E621Cul;

        struct malloc_payload {
            PyObject *arr;
            unsigned long magic;
        };

        BOOST_STATIC_ASSERT(sizeof(unsigned long) == 8);
        BOOST_STATIC_ASSERT(sizeof(malloc_payload) <= 16);

        // A Python object that will call free when its reference count reaches 0
        class malloc_release {
        public:
            malloc_release(){}
            malloc_release(void* p) { set_ptr(p); }

            void set_ptr(void *p) {
                ptr.reset(p, std::ptr_fun(free));
            }

        private:
            std::shared_ptr<void> ptr;
        };

        PyObject *get_malloc_payload(void *ptr) {
            // Retrieve the original object
            void *mem = ((char*) ptr) - 16;
            // Check the payload
            malloc_payload *payload = (malloc_payload *) mem;
            if (payload->magic != DATA_MAGIC) {
                // Abort execution rather than throw an exception
                fprintf(stderr, "Abort: get_malloc_payload: ptr was not allocated by numpy_malloc\n");
                abort();
            }
            return payload->arr;
        }

        bp::object make_temporary_array_view(N_Vector vec) {
            if (!vec) {
                throw std::invalid_argument("make_temporary_array_view: vec is NULL");
            }
            npy_intp dims[] = { npy_intp(NV_LENGTH_S(vec)) };
            PyObject *arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, NV_DATA_S(vec));
            if (!arr) {
                bp::throw_error_already_set();
            }
            return bp::object(bp::handle<>(arr));
        }
    }

    extern "C" void * numpy_malloc(size_t len, size_t el_size) {
        if (!len) return NULL;
        ASSERT(el_size == sizeof(double));

        // Allocate memory aligned to 16 bytes
        void *mem = 0;
        int res = posix_memalign(&mem, 16, 16 + len*el_size);
        if (res != 0 || !mem) return 0;
        malloc_payload *payload = (malloc_payload *) mem;
        void *array_data = ((char*) mem) + 16;

        // This function may be called with GIL unlocked, so re-acquire the GIL if necessary
        finmag::util::scoped_gil_ensure lock;

        // Create a Python object handle to keep hold of mem
        bp::object mem_release = bp::object(malloc_release(mem));
        // Create a numpy array using array_data for storage and mem_release as base
        // See http://blog.enthought.com/python/numpy-arrays-with-pre-allocated-memory/
        // for rationale and explanation
        npy_intp dims[] = { npy_intp(len) };
        PyObject * arr = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, array_data);
        // Note that return 0 will release the memory pointed to by mem
        if (!arr) return 0;
        // incref mem_release and save it in arr
	py_array_set_base((PyArrayObject *) arr, bp::handle<>(bp::borrowed(mem_release.ptr())).release());

        // set up the payload
        payload->arr = arr;
        payload->magic = DATA_MAGIC;

        // return the array data
        return array_data;
    }

    extern "C" void numpy_free(void *ptr) {
        if (!ptr) return;

        // Retrieve the original object
        void *mem = ((char*)ptr) - 16;
        // Check the payload
        malloc_payload *payload = (malloc_payload *) mem;
        if (payload->magic != DATA_MAGIC) {
            // Abort execution rather than throw an exception
            fprintf(stderr, "Abort: numpy_free: ptr was not allocated by numpy_malloc\n");
            abort();
        }

        // This function may be called with GIL unlocked, so re-acquire the GIL if necessary
        finmag::util::scoped_gil_ensure lock;
        // Retrieve the original numpy array object and decrease its reference count
        // If the reference count goes to 0, this will free the memory pointed by mem
        bp::handle<> handle(payload->arr);
    }

    array_nvector::array_nvector(const np_array<double> &arr): vec(0), arr(arr) {
        // Create an N_Vector using data as storage
        vec = N_VMake_Serial(arr.size(), arr.data());
        if (!vec) throw std::runtime_error("N_VMake_Serial returned NULL");
    }

    bp::object nvector_to_array_object(N_Vector vec) {
        return make_temporary_array_view(vec);
    }

    np_array<double> nvector_to_array(N_Vector vec) {
         return bp::extract<np_array<double> >(nvector_to_array_object(vec));
    }

    void register_numpy_malloc() {
        bp::class_<malloc_release>("_malloc_release", bp::init<>());
    }
}}
