#include "finmag_includes.h"

#include <boost/test/unit_test.hpp>

#include "my_boundary_mesh.h"

BOOST_AUTO_TEST_CASE(test_numpy_malloc)
{
    using namespace dolfin;

    UnitCube mesh(1, 1, 1);
    MyBoundaryMesh boundary_mesh(mesh);
    Point p(0.5, 0.5, 0.5);
    for (unsigned i = 0; i < boundary_mesh.num_cells(); i++) {
        Cell cell(boundary_mesh, i);
        Point p1 = boundary_mesh.geometry().point(cell.entities(0)[0]);
        Point p2 = boundary_mesh.geometry().point(cell.entities(0)[1]);
        Point p3 = boundary_mesh.geometry().point(cell.entities(0)[2]);
        Point v1 = p2 - p1;
        Point v2 = p3 - p1;
        Point n  = v1.cross(v2);
        printf("%d: %d,%d,%d %g\n", i, cell.entities(0)[0], cell.entities(0)[1], cell.entities(0)[2], n.dot(p1 - p));
    }
}
