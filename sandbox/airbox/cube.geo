algebraic3d

solid cube = orthobrick(0, 0, 0; 20, 20, 20) -maxh = 5.0;
solid box = orthobrick(-120, -120, -120; 140, 140, 140) -maxh = 15.0;
solid air = box and not cube;

tlo cube;
tlo air;
