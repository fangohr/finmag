algebraic3d

# the film has dimensions 1000 x 42 x 5 nm
# where x in [0, 1000], y in [-21, 21] and z in [-2.5, 2.5]

solid film = 
    orthobrick (0, -21, -2.5; 1000, 21, 2.5)
    -maxh = 4.0;

# a current will be applied near the left edge of the film in a region
# which corresponds to a circular contact with radius 20 nm

solid contact =
    cylinder (225, 0, -2.5; 225, 0, 2.5; 20)
    and plane (0, 0, -2.5; 0, 0, -1)
    and plane (0, 0,  2.5; 0, 0,  1)
    and (not film)
    -maxh = 4.0;

tlo contact;
tlo film;
