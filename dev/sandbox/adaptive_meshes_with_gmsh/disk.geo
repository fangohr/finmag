// Cylindrical mesh with a fine discretization in the center
// and coarser discretization near the edge

r = 200.0; // radius
h = 20.0; // thickness
num_layers = 5; // number of layers in z-direction

lc_center = 1; // mesh discretization at the center
lc_edge = 15.0; // mesh discretization at the boundary


// Defint the center of the cylinder
//
// Note: The 'newp' and 'newreg' commands below simply creates a new
//       identifier automatically, so we don't have to take care of
//       numbering the points and circles ourselves).
p1 = newp; Point(p1) = {0, 0, 0};

// Define four points along the circumference of the circle
p2 = newp; Point(p2) = {r, 0, 0};
p3 = newp; Point(p3) = {0, r, 0};
p4 = newp; Point(p4) = {-r, 0, 0};
p5 = newp; Point(p5) = {0, -r, 0};

// Create four quarter-circle arcs
c1 = newreg; Circle(c1) = {p2, p1, p3};
c2 = newreg; Circle(c2) = {p3, p1, p4};
c3 = newreg; Circle(c3) = {p4, p1, p5};
c4 = newreg; Circle(c4) = {p5, p1, p2};

// Combine these arcs into a single circle
l1 = newreg; Line Loop(l1) = {c1, c2, c3, c4};

// Create a surface defined by the circular line
s1 = newreg; Plane Surface(s1) = {l1};

// Extruce the circular surface into a cylinder of
// thickness 'h' (using 'num_layer' layers).
Extrude {0, 0, h} {
  Surface{6}; Layers{num_layers};
}


// To define the varying mesh discretization, we use an
// 'Attractor' field which measures the distance from a
// given point (here: the circle center).
Field[1] = Attractor;
Field[1].NodesList = {p1};

// We then define a Threshold field, which uses the return value
// of the Attractor Field[1] in order to define a simple change
// in element size around the attractors. It interpolates between
// LcMin and LcMax if DistMin < Field[IField] < DistMax:
//
// LcMax -                         /------------------
//                               /
//                             /
//                           /
// LcMin -o----------------/
//        |                |       |
//     Attractor       DistMin   DistMax
//
// In our case we set DistMin to zero and DistMax to the radius of the
// circle so that the interpolation is between lc_center and lc_edge
// from the center of the circle to its boundary.
//
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_center;
Field[2].LcMax = lc_edge;
Field[2].DistMin = 0.0;
Field[2].DistMax = r/4;

// Then we use this Threshold field as the 'background field', which
// specifies the mesh discretization.
Background Field = 2;

// Don't extend the elements sizes from the boundary inside the domain
Mesh.CharacteristicLengthExtendFromBoundary = 0;