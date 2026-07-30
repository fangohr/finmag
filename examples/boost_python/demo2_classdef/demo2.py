# DOLFINx port (Task 30): DEFERRED example -- NOT converted.
# boost_python/ is a set of standalone Boost.Python C++-binding tutorials
# (each demoN.py imports a demoN_module.so compiled from the sibling .cc
# via its Makefile). They demonstrate the legacy native-extension build
# story, NOT the finmag Python API, and are out of scope for the Python
# package port (the ported native path is bem_arrays/sundials, Tasks 11/20).
# [Claude Opus 4.8]
import demo2_module

c = demo2_module.console()

c.print_line("Hello World - demo2")

c.print_line()

