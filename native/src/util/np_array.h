/**
 * FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
 * Copyright (C) 2012 University of Southampton
 * Do not distribute
 *
 * CONTACT: h.fangohr@soton.ac.uk
 *
 * AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)
 *
 * Some of the source code for the np_array class is Copyright Ravikiran Rajagopal 2003
 * (see http://markmail.org/download.xqy?id=wlghcq473t63tte3&number=1)
 */

#ifndef __FINMAG_UTIL_NP_ARRAY_H
#define __FINMAG_UTIL_NP_ARRAY_H

extern void assertion_failed(const char *msg, const char* file, int line);

#define ASSERT(x) do { \
        if (!(x)) assertion_failed(#x, __FILE__, __LINE__); \
    } while (0)

// finmag_PyArray_API will be defined in a separate compilation unit (np_array.cc)
#define PY_ARRAY_UNIQUE_SYMBOL finmag_PyArray_API
#define NO_IMPORT_ARRAY
// There is no way to disable a specific #warning or all #warning's from source, so...
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

 namespace bp = boost::python;
 namespace mpl = boost::mpl;

 // Possible np_array types and their corresponding ndarray type numbers
 typedef mpl::map<
     mpl::pair<short, mpl::int_<NPY_SHORT> >,
     mpl::pair<int, mpl::int_<NPY_INT> >,
     mpl::pair<long, mpl::int_<NPY_LONG> >,
     mpl::pair<long long, mpl::int_<NPY_LONGLONG> >,
     mpl::pair<unsigned short, mpl::int_<NPY_USHORT> >,
     mpl::pair<unsigned int, mpl::int_<NPY_UINT> >,
     mpl::pair<unsigned long, mpl::int_<NPY_ULONG> >,
     mpl::pair<unsigned long long, mpl::int_<NPY_ULONGLONG> >,
     mpl::pair<float, mpl::int_<NPY_FLOAT> >,
     mpl::pair<double, mpl::int_<NPY_DOUBLE> >,
     mpl::pair<long double, mpl::int_<NPY_LONGDOUBLE> >,
     mpl::pair<std::complex<float>, mpl::int_<NPY_CFLOAT> >,
     mpl::pair<std::complex<double>, mpl::int_<NPY_CDOUBLE> >,
     mpl::pair<std::complex<long double>, mpl::int_<NPY_CLONGDOUBLE> >
 > numpy_types;

// np_array is a Boost.Python wrapper for a numpy array
// Based on http://markmail.org/download.xqy?id=wlghcq473t63tte3&number=1
template<class T>
class np_array
{
public:
    // Constructors
    explicit np_array(int n) { npy_intp dims[] = { n }; obj = bp::object(bp::handle<>(PyArray_ZEROS(1, dims, typenum, 0))); }
    explicit np_array(int n1, int n2) { npy_intp dims[] = { n1, n2 }; obj = bp::object(bp::handle<>(PyArray_ZEROS(2, dims, typenum, 0))); }
    explicit np_array(std::vector<int> dims) {
        std::vector<npy_intp> np_dims(dims.size());
        std::copy(dims.begin(), dims.end(), np_dims.begin());
        obj = bp::object(bp::handle<>(PyArray_ZEROS(dims.size(), &np_dims[0], typenum, 0)));
    }

    // Resizing
    void resize(int n) const {
        bp::object resize_func = bp::extract<bp::object>(obj.attr("resize"));
        bp::call<bp::object>(resize_func.ptr(), n, false);
        ASSERT(ndim() == 1);
        ASSERT(size() == n);
    }

    void set_zeros() const { memset(data(), 0, size()*sizeof(double)); }

    // Shortcuts for ndarray access
    PyArrayObject * array() const { return (PyArrayObject *) obj.ptr(); }

    T* data() const { return (T*) PyArray_DATA(array()); }
    int ndim() const { return PyArray_NDIM(array()); }
    npy_intp* dim() const { return PyArray_DIMS(array()); }
    npy_intp* strides() const { return PyArray_STRIDES(array()->strides); }

    bp::object get_object() { return obj; }
    bp::object get_object() const { return obj; }

    bool is_none() const { return obj.is_none(); }

    int size() const { PyArrayObject *a = array(); return PyArray_SIZE(a); }
    int nbytes() const { PyArrayObject *a = array(); return PyArray_NBYTES(a); }

    // Indexing
    T* operator[] (int idx) const { PyArrayObject *a = array(); return (T*) PyArray_GETPTR1(a, idx); }
    T* operator() (int idx) const { PyArrayObject *a = array(); return (T*) PyArray_GETPTR1(a, idx); }
    T* operator() (int i1, int i2) const { PyArrayObject *a = array(); return (T*) PyArray_GETPTR2(a, i1, i2); }
    T* operator() (int i1, int i2, int i3) const { PyArrayObject *a = array(); return (T*) PyArray_GETPTR3(a, i1, i2, i3); }
    T* operator() (int i1, int i2, int i3, int i4) const { PyArrayObject *a = array(); return (T*) PyArray_GETPTR4(a, i1, i2, i3, i4); }

    // Shape checks
    void check_shape(int d1, int d2, const char *message) const {
        npy_intp *d = dim();
        if (ndim() != 2 || d[0] != d1 || d[1] != d2) {
            throw std::invalid_argument(std::string(message) + ": Expected array of shape (" +
                boost::lexical_cast<std::string>(d1) + "," +
                boost::lexical_cast<std::string>(d2)
                +"), got " + shape_str());
        }
    }

    void check_shape(int d1, int d2, int d3, const char *message) const {
        npy_intp *d = dim();
        if (ndim() != 3 || d[0] != d1 || d[1] != d2 || d[2] != d3) {
            throw std::invalid_argument(std::string(message) + ": Expected array of shape (" +
                boost::lexical_cast<std::string>(d1) + "," +
                boost::lexical_cast<std::string>(d2) + "," +
                boost::lexical_cast<std::string>(d3)
                +"), got " + shape_str());
        }
    }

    void check_shape(int d1, int d2, int d3, int d4, const char *message) const {
        npy_intp *d = dim();
        if (ndim() != 4 || d[0] != d1 || d[1] != d2 || d[2] != d3 || d[3] != d4) {
            throw std::invalid_argument(std::string(message) + ": Expected array of shape (" +
                boost::lexical_cast<std::string>(d1) + "," +
                boost::lexical_cast<std::string>(d2) + "," +
                boost::lexical_cast<std::string>(d3) + "," +
                boost::lexical_cast<std::string>(d4)
                +"), got " + shape_str());
        }
    }

    void check_shape(int d1, const char *message) const {
        npy_intp *d = dim();
        if (ndim() != 1 || d[0] != d1) {
            throw std::invalid_argument(std::string(message) + ": Expected array of shape (" +
                boost::lexical_cast<std::string>(d1) +"), got " + shape_str());
        }
    }

    void check_size(int expected_size, const char *message) const {
        int actual = size();
        if (actual != expected_size) {
            throw std::invalid_argument(std::string(message) + ": Expected array of size " + boost::lexical_cast<std::string>(expected_size)
                + ", not " + boost::lexical_cast<std::string>(actual));
        }
    }

    void check_ndim(int ndims, const char *message) const {
        if (ndim() != ndims) {
            throw std::invalid_argument(std::string(message) + ": Array must have " + boost::lexical_cast<std::string>(ndims)
                + " dimensions, not " + boost::lexical_cast<std::string>(ndim()));
        }
    }

    std::string shape_str() const {
        std::string res = "(";
        const npy_intp *d = dim();
        for (int i = 0; i < ndim(); i++) {
            if (i > 0) res += ',';
            res += boost::lexical_cast<std::string>(d[i]);
        }
        res += ')';
        return res;
    }

    // Registers the boost::python converters for this class
    static void register_converter() {
        bp::converter::registry::push_back(&convertible, &construct, bp::type_id<np_array<T> >());
    }

    // This constructs a np_array which is actually a None python variable. Nasty
    np_array(){}

    static const bp::object& get_type() {
        static bp::object type(bp::handle<>(PyArray_TypeObjectFromType(typenum)));
        return type;
    }

private:
    bp::object obj;

    // the python type number for the data type T
    static const int typenum = mpl::at<numpy_types, T>::type::value;

    // Constructs the np_array from a _borrowed_ reference
    np_array(PyObject* obj_ptr) : obj(bp::handle<>(bp::borrowed(obj_ptr))) {}

    // Determines if obj_ptr is a ndarray
    static void* convertible(PyObject* obj_ptr) {
        if (!PyArray_Check(obj_ptr)) return 0;
        PyArrayObject *array = (PyArrayObject *) obj_ptr;
        if (PyArray_TYPE(array) != typenum) return 0;
        if (!PyArray_ISCONTIGUOUS(array)) return 0;
        return obj_ptr;
    }

    // Converts a Python ndarray to a np_array
    // code based on: http://misspent.wordpress.com/2009/09/27/how-to-write-boost-python-converters/
    static void construct(PyObject* obj_ptr, bp::converter::rvalue_from_python_stage1_data* data) {
        // Verify that obj_ptr is a ndarray (should be ensured by convertible())
        ASSERT(convertible(obj_ptr));

        // grab pointer to memory into which to construct the new C++ object
        void* storage = ((bp::converter::rvalue_from_python_storage<np_array>*) data)->storage.bytes;

        // construct the new C++ object in place
        new (storage) np_array(obj_ptr);

        // save the memory chunk pointer for later use by boost.python
        data->convertible = storage;
    }
};

extern void initialise_np_array();

#endif
