set multiplot layout 2, 3 title "finmag/nmag comparison"

set xtic 0, 10e-12, 10e-12
set xlabel "time (s)"
set ylabel "polarisation"

set title "m_x"
set yrange [0.94 : 1.00]
set ytic 0.95, 0.05, 1.00
plot \
    "averages_ref.txt" using 1:2 title "nmag" with lines lc rgbcolor "black", \
    "averages.txt" using 1:2 title "finmag" with lines lc rgbcolor "red"

set nokey

set ylabel ""
set title "m_y"
set yrange [0.10 : 0.11]
set ytic 0.10, 0.005, 0.11
plot \
    "averages_ref.txt" using 1:3 title "nmag" with lines lc rgbcolor "black", \
    "averages.txt" using 1:3 title "finmag" with lines lc rgbcolor "red"

set title "m_z"
set yrange [-0.0002 : 0.0005]
set ytic 0.00, 0.0005, 0.0005
plot \
    "averages_ref.txt" using 1:4 title "nmag" with lines lc rgbcolor "black", \
    "averages.txt" using 1:4 title "finmag" with lines lc rgbcolor "red"

set ylabel "difference"

set title "dx"
set yrange [0.00 : 0.01]
set ytic 0.00, 0.005, 0.01 
plot \
    "< paste averages_ref.txt averages.txt" using 1:(abs($2-$6)) with lines lc rgbcolor "red"

set ylabel ""
set title "dy"
set yrange [0.00 : 0.0025]
set ytic 0.00, 0.001, 0.0025
plot \
    "< paste averages_ref.txt averages.txt" using 1:(abs($3-$7)) with lines lc rgbcolor "red"


set title "dz"
set yrange [0.00 : 0.0004]
set ytic 0.00, 0.0002, 0.0004
plot \
    "< paste averages_ref.txt averages.txt" using 1:(abs($4-$8)) with lines lc rgbcolor "red"

pause -1
