#include <boost/python.hpp>

namespace bp = boost::python;

void print_line(std::string message)
{
    std::cout << message << std::endl;
}

// Name of the module in () has to be the same as the name of the compiled .so file that gets imported
BOOST_PYTHON_MODULE(demo1_module)
{
    bp::scope().attr("__doc__") = "Boost.Python Demo #1 - Hello World";

    bp::def("print_line", &print_line);
}
