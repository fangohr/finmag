algebraic3d

solid permalloy = sphere(0, 0, 0; 1);
solid box = orthobrick(-11, -11, -11; 11, 11, 11) -maxh = 5.0;
solid air = box and not permalloy;

tlo permalloy;
tlo air;
