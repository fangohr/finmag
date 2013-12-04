// Cylindrical mesh with a fine discretization in the center
// and coarser discretization near the edge

r = 50;
h = 0.001;

maxh_edge = 10.0;
maxh_center = 10.0;

Point(1) = {0,  0, 0, maxh_center};

Point(2) = {r, 0, 0, 2.0};
Point(3) = {0, r, 0, 2.0};
Point(4) = {-r, 0, 0, maxh_edge};
Point(5) = {0, -r, 0, maxh_edge};

c1 = newreg; Circle(c1) = {2, 1, 3};
c2 = newreg; Circle(c2) = {3, 1, 4};
c3 = newreg; Circle(c3) = {4, 1, 5};
c4 = newreg; Circle(c4) = {5, 1, 2};

l1 = newreg; Line Loop(l1) = {c1, c2, c3, c4};

s1 = newreg; Plane Surface(s1) = {l1};

Extrude {0, 0, h} {
  Surface{6}; Layers{1};
}

Field[1] = MathEval;
Field[1].F = "10 * Sqrt(x*x+y*y)/90 + 1 * (1 - Sqrt(x*x+y*y)/90)";

Background Field = 1;

// Don't extend the elements sizes from the boundary inside the domain
Mesh.CharacteristicLengthExtendFromBoundary = 0;