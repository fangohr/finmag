algebraic3d

# The film described in figures 5 and 6, page 7 and section 5, page 8 is
# 440 nm * 440 nm * 5 nm. We'll put the center of the film at 0,0,0. However,
# the film doesn't have an area of 600 nm^2, as stated in Sec. 5 p. 8.

solid surface = orthobrick (-220, -220, -2.5; 220, 220, 2.5) -maxh = 2.0;

solid contact1 = cylinder (-62.5, 0, -2.5; -62.5, 0, 2.5; 10)
                    and plane (0, 0, -2.5; 0, 0, -1)
                    and plane (0, 0,  2.5; 0, 0,  1) -maxh = 4.0;

solid contact2 = cylinder (62.5, 0, -2.5; 62.5, 0, 2.5; 10)
                    and plane (0, 0, -2.5; 0, 0, -1)
                    and plane (0, 0,  2.5; 0, 0,  1) -maxh = 4.0;

solid aroundcontact1 = cylinder (-62.5, 0, -2.5; -62.5, 0, 2.5; 62)
                    and plane (0, 0, -2.5; 0, 0, -1)
                    and plane (0, 0,  2.5; 0, 0,  1)
                    and not contact1 -maxh = 3.0;

solid aroundcontact2 = cylinder (62.5, 0, -2.5; 62.5, 0, 2.5; 62)
                    and plane (0, 0, -2.5; 0, 0, -1)
                    and plane (0, 0,  2.5; 0, 0,  1)
                    and not contact2 -maxh= 3.0;

solid film = surface and (not contact1) and (not contact2) and (not aroundcontact1) and (not aroundcontact2); 

tlo contact1;
tlo aroundcontact1;
tlo contact2;
tlo aroundcontact2;
tlo film;
