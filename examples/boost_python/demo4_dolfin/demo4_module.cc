#include "finmag_includes.h"
#include "../../../native/src/util/swig_dolfin.h"

int get_num_vertices(const std::shared_ptr<dolfin::Mesh> &mesh_ptr) {
    dolfin::Mesh &mesh = *mesh_ptr;

    return mesh.num_vertices();
}

BOOST_PYTHON_MODULE(demo4_module)
{
    finmag::util::register_dolfin_swig_converters();

    boost::python::scope().attr("__doc__") = "Boost.Python Demo #4 - Dolfin";

    boost::python::def("get_num_vertices", &get_num_vertices);
}
