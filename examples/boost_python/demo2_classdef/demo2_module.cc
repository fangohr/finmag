#include <boost/python.hpp>

namespace bp = boost::python;

class console
{
public:
    void print_line(std::string message) {
        std::cout << message << std::endl;
    }
};

BOOST_PYTHON_MODULE(demo2_module)
{
    bp::scope().attr("__doc__") = "Boost.Python Demo #2 - Class Definition";

    using namespace bp;

    class_<console>("console", "console class docstring", init<>())
        .def("print_line", &console::print_line, args("message"), "prints a message via the standard output stream")
    ;
}
