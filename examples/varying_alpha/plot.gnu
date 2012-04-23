set terminal postscript enhanced eps color
set pm3d map 
set palette defined ( 0 '#000090',\
                      1 '#000fff',\
                      2 '#0090ff',\
                      3 '#0fffee',\
                      4 '#90ff70',\
                      5 '#ffee00',\
                      6 '#ff7000',\
                      7 '#ee0000',\
                      8 '#7f0000')
set out "dispersion.eps"
set xlabel "k_x (nm^{-1})"
set ylabel "Frequency (GHz)"

set xrange [-0.5:0.5]
set yrange [0:50]

set size 1,0.6

splot 'dispersion.dat' using 1:2:3  t ""
