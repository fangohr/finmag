algebraic3d

# film described in figures 5 and 6, page 7 and section 5, page 8 is
# 440 nm * 440 nm * 5 nm. We'll put the center of the film at 0,0,0.
solid film = orthobrick (-220, -220, -2.5; 220, 220, 2.5) -maxh = 6.0;
tlo film;
