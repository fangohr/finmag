algebraic3d

solid bar = orthobrick (-540, -15, -1; 540, 15, 1);

solid hole1 = orthobrick (-544.5, 12, -1.1; -535.5, 15.1, 1.1);
solid hole2 = orthobrick (-544.5, -15.1, -1.1; -535.5, -12, 1.1);

solid holes1 = multitranslate (18, 0, 0; 60; hole1);
solid holes2 = multitranslate (18, 0, 0; 60; hole2);

solid main = bar and not holes1 and not holes2;

tlo main -maxh=2;