### a three-quarter sphere
### demonstrates basic CSG boolean operations

algebraic3d
solid smallsphere = sphere(0,0,0;5)  -maxh=1;
solid largesphere = sphere(0,0,0;10)  -maxh=1;
solid smallsphere2 = sphere(15,0,0;5)  -maxh=1;

solid shell = largesphere and not smallsphere;
tlo shell;
tlo smallsphere;