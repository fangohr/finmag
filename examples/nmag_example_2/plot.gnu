set terminal png
set output "agreement_averages.png"
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

set xlabel "time (ns)"
set xrange [0:0.3]
set xtic 0, 0.1, 0.3
set format x "%.0g"

set ylabel "ratio (1)"
set yrange [0.98:1.02]
set ytic 0.98, 0.02, 1.02
set format y "%.2f"

set nokey
plot \
    "< paste averages.txt averages_ref.txt" using ($1*1e9):($2/$6) with points lc rgbcolor "red", \
    "< paste averages.txt averages_ref.txt" using ($1*1e9):($3/$7) with points lc rgbcolor "blue", \
    "< paste averages.txt averages_ref.txt" using ($1*1e9):($4/$8) with points lc rgbcolor "green"

#
# UPPER GRAPH - Magnetisations.
# 

set origin 0.0, 0.4
set size   1.0, 0.55 # some space on top for the page title.

set tmargin 1
set bmargin 0

set xlabel ""
set format x ""

set ylabel "magnetisation (1)"
set yrange [-0.2:1]
set ytic 0.0, 0.5, 1.0
set format y "%.2f"

set key
plot \
    "averages_ref.txt" using ($1*1e9):2 title "m_x (nmag)" with lines lc rgbcolor "red", \
    "averages_ref.txt" using ($1*1e9):3 title "m_y" with lines lc rgbcolor "blue", \
    "averages_ref.txt" using ($1*1e9):4 title "m_z" with lines lc rgbcolor "green", \
    "averages.txt" using ($1*1e9):2 title "m_x (finmag)" with points lc rgbcolor "red", \
    "averages.txt" using ($1*1e9):3 title "m_y" with points lc rgbcolor "blue", \
    "averages.txt" using ($1*1e9):4 title "m_z" with points lc rgbcolor "green"

unset multiplot 
