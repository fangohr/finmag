algebraic3d

solid disk =
    (cylinder (0.0, 0.0, -1.0; 0.0, 0.0, 1.0; 10)
        and plane (0, 0, -10.0; 0, 0, -1)
        and plane (0, 0,  10.0; 0, 0,  1)) -maxh = 1.0;

solid film =
    orthobrick ( -30.0, -30.0, -10.0 ; 30.0, 30.0, 10.0 )
    and (not disk) -maxh = 5.0;

tlo disk;
tlo film;
