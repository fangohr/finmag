algebraic3d

# surface described in figures 5 and 6, page 7 and section 5, page 8 is
# 440 nm * 440 nm * 5 nm. We'll put the center of the surface at 0,0,0.
solid surface = orthobrick (-220, -220, -2.5; 220, 220, 2.5) -maxh = 5.0;

solid contact1 = cylinder (-62.5, 0, -2.5; -62.5, 0, 2.5; 10)
                    and plane (0, 0, -2.5; 0, 0, -1)
                    and plane (0, 0,  2.5; 0, 0,  1) -maxh = 5.0;
solid contact2 = cylinder (62.5, 0, -2.5; 62.5, 0, 2.5; 10)
                    and plane (0, 0, -2.5; 0, 0, -1)
                    and plane (0, 0,  2.5; 0, 0,  1) -maxh = 5.0;
solid film = surface and not contact1 and not contact2; 

tlo contact1;
tlo contact2;
tlo film;
