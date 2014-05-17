#include "finmag_includes.h"
#include "../../../native/src/util/swig_dolfin.h"

int get_num_vertices(const boost::shared_ptr<dolfin::Mesh> &mesh_ptr) {
    dolfin::Mesh &mesh = *mesh_ptr;

    return mesh.num_vertices();
}

int get_vector_local_size(const boost::shared_ptr<dolfin::GenericVector> &vector_ptr) {
	dolfin::GenericVector &v = *vector_ptr;

	return v.local_size();
}

BOOST_PYTHON_MODULE(demo5_module)
{
    finmag::util::register_dolfin_swig_converters();

    boost::python::scope().attr("__doc__") = "Boost.Python Demo #5 - Dolfin";

    boost::python::def("get_num_vertices", &get_num_vertices);

    boost::python::def("get_vector_local_size", &get_vector_local_size);
}
