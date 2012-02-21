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
set xrange [0:1e-9]
set xtic 0, 2e-10, 8e-10
set format x "%.0g"

set ylabel "ratio"
set yrange [0.5:1.5]
set ytic 0.6, 0.4, 1.4
set format y "%.1f"

set nokey
plot \
    "< paste data_M.txt data_M_dolfin.txt" using 1:($2/$6), \
    "< paste data_M.txt data_M_dolfin.txt" using 1:($3/$7), \
    "< paste data_M.txt data_M_dolfin.txt" using 1:($4/$8)

#
# UPPER GRAPH - Magnetisations.
# 

set origin 0.0, 0.4
set size   1.0, 0.55 # some space on top for the page title.

set tmargin 1
set bmargin 0

set xlabel ""
set format x ""

set ylabel "magnetisation (A/m)"
set yrange [-5e5:1e6]
set ytic -4e5, 4e5, 8e5
set format y "%.1g"

set key
plot \
    "data_M.txt" using 1:2 title "M_x (nmag)" with lines lc rgbcolor "red", \
    "data_M.txt" using 1:3 title "M_y" with lines lc rgbcolor "blue", \
    "data_M.txt" using 1:4 title "M_z" with lines lc rgbcolor "green", \
    "data_M_dolfin.txt" using 1:2 title "M_x (finmag)" with points lc rgbcolor "red", \
    "data_M_dolfin.txt" using 1:3 title "M_y" with points lc rgbcolor "blue", \
    "data_M_dolfin.txt" using 1:4 title "M_z" with points lc rgbcolor "green"

unset multiplot 
