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
#include "nvector_custom_malloc.h"

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
        static const unsigned long NVEC_MAGIC = 0x24A3D2A040B0D56Ful;

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
            boost::shared_ptr<void> ptr;
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

        struct nvector_extended_content {
            _N_VectorContent_Serial content;
            unsigned long magic;
            PyObject *array_data;
        };

        PyObject *& get_nvec_array_data(N_VectorContent_Serial vec) {
            nvector_extended_content *c = (nvector_extended_content*) vec;
            if (c->magic != NVEC_MAGIC) {
                // Abort execution rather than throw an exception
                fprintf(stderr, "Abort: get_nvec_array_data: pointer was not allocated by numpy_nvec_malloc\n");
                abort();
            }
            return c->array_data;
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

    extern "C" N_VectorContent_Serial numpy_nvec_malloc() {
        // allocate memory
        nvector_extended_content *c = (nvector_extended_content*) malloc(sizeof(nvector_extended_content));
        if (!c) return 0;
        // fill out magic and zero out numpy array pointer
        c->magic = NVEC_MAGIC;
        c->array_data = 0;
        // the allocated memory pointer must be the same as the contents (c->content) since it's the first field
        ASSERT((void*)c == (void*)&c->content);
        return &c->content;
    }

    extern "C" void numpy_nvec_free(N_VectorContent_Serial vec) {
        nvector_extended_content *c = (nvector_extended_content*) vec;
        if (c->magic != NVEC_MAGIC) {
            // Abort execution rather than throw an exception
            fprintf(stderr, "Abort: numpy_nvec_free: ptr was not allocated by numpy_nvec_malloc\n");
            abort();
        }
        free(vec);
    }

    array_nvector::array_nvector(const np_array<double> &arr): vec(0), arr(arr) {
        // Create an N_Vector using data as storage
        vec = N_VMake_Serial(arr.size(), arr.data());
        if (!vec) throw std::runtime_error("N_VMake_Serial returned NULL");
        // fill in the pointer to the original numpy for this NVector
        PyObject *&array_data = get_nvec_array_data(NV_CONTENT_S(vec));
        array_data = arr.get_object().ptr();
    }

    bp::object nvector_to_array_object(N_Vector vec) {
        if (!vec) throw std::invalid_argument("nvector_to_array: vec is NULL");
        // First, check if the array_data is filled in
        PyObject *&array_data = get_nvec_array_data(NV_CONTENT_S(vec));
        if (array_data) {
            // array_data is set, use the pointer
            // for safety, check that the data pointer in the numpy array is the same as in the NVector
            ASSERT((void*) PyArray_BYTES((PyArrayObject*) array_data) == (void*) NV_DATA_S(vec));
            return bp::object(bp::handle<>(bp::borrowed(array_data)));
        }

        // if array_data is not set but NV_OWN_DATA_S is false, then we cannot retrieve the original pointer (if any!)
        if (NV_OWN_DATA_S(vec) == FALSE) {
            throw std::invalid_argument("nvector_to_array: NVector was not created from a numpy array but has OWN_DATA=false, cannot convert");
        }

        // Beyond this point, we should be able to retrieve the original pointer unless the data pointer has been tampered with
        // (someone has assigned to NV_DATA_S(vec) directly, or called N_VSetArrayPointer_Serial)

         // Retrieve the original pointer
         PyObject *data = (PyObject *) get_malloc_payload(NV_DATA_S(vec));
         return bp::object(bp::handle<>(bp::borrowed(data)));
    }

    np_array<double> nvector_to_array(N_Vector vec) {
         return bp::extract<np_array<double> >(nvector_to_array_object(vec));
    }

    void register_numpy_malloc() {
        set_nvector_custom_allocators(numpy_malloc, numpy_free, numpy_nvec_malloc, numpy_nvec_free);

        bp::class_<malloc_release>("_malloc_release", bp::init<>());
    }
}}
