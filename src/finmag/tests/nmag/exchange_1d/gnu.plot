set lmargin 12
set rmargin 2
set grid

set multiplot title "Agreement of finmag and nmag"

#
# LOWER GRAPH - Ratios.
#

set origin 0.0, 0.0
set size   1.0, 0.4

set tmargin 0
set bmargin 3

set xlabel "time (s)"
set xrange [0:1e-10]
set xtic 0, 2e-10, 8e-10
set format x "%.0g"

set ylabel "ratio"
set yrange [0.0 : 2.0]
set ytic 0.0, 1.0, 2.0
set format y "%.1f"

set nokey
plot \
    "< paste averages_ref.txt averages.txt" using 1:($2/$6) with points lc rgbcolor "red", \
    "< paste averages_ref.txt averages.txt" using 1:($3/$7) with points lc rgbcolor "blue", \
    "< paste averages_ref.txt averages.txt" using 1:($4/$8) with points lc rgbcolor "green"

#
# UPPER GRAPH - Magnetisations.
# 

set origin 0.0, 0.4
set size   1.0, 0.55 # some space on top for the page title.

set tmargin 1
set bmargin 0

set xlabel ""
set format x ""

set ylabel "polarisation"
set yrange [-0.1:1]
set ytic -0.5 0.5 1
set format y "%.1g"

set key
plot \
    "averages_ref.txt" using 1:2 title "M_x (nmag)" with lines lc rgbcolor "red", \
    "averages_ref.txt" using 1:3 title "M_y" with lines lc rgbcolor "blue", \
    "averages_ref.txt" using 1:4 title "M_z" with lines lc rgbcolor "green", \
    "averages.txt" using 1:2 title "M_x (finmag)" with points lc rgbcolor "red", \
    "averages.txt" using 1:3 title "M_y" with points lc rgbcolor "blue", \
    "averages.txt" using 1:4 title "M_z" with points lc rgbcolor "green"

unset multiplot 

pause -1
