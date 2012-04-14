set macro

set terminal postscript
set output "| ps2pdf - diff.pdf"

# sources
ref = '"averages_ref.txt"'              # the nmag data
new = '"averages.txt"'                  # the finmag data
refnew = '"< paste averages_ref.txt averages.txt"'

# styles
nmag = 'with lines lc rgbcolor "black"' # for the nmag curves
finmag = 'with lines lc rgbcolor "red"' # for the finmag curves
diff = 'with lines lc rgbcolor "red"'   # for difference curves

set multiplot layout 2, 3 title "finmag/nmag comparison"

set xtic 0, 10e-12, 10e-12
set xlabel "time (s)"
set ylabel "polarisation"

set title "m_x"
#set yrange [0.94 : 1.00]
#set ytic 0.95, 0.05, 1.00
plot \
    @ref using 1:2 title "nmag" @nmag, \
    @new using 1:2 title "finmag" @finmag

set nokey

set ylabel ""
set title "m_y"
#set yrange [0.10 : 0.11]
#set ytic 0.10, 0.005, 0.11
plot \
    @ref using 1:3 title "nmag" @nmag, \
    @new using 1:3 title "finmag"  @finmag

set title "m_z"
#set yrange [-0.0002 : 0.0005]
#set ytic 0.00, 0.0005, 0.0005
plot \
    @ref using 1:4 title "nmag" @nmag, \
    @new using 1:4 title "finmag" @finmag

set ylabel "difference"

set title "dx"
#set yrange [0.00 : 0.01]
#set ytic 0.00, 0.005, 0.01 
plot @refnew using 1:(($2-$6)) @diff 

set ylabel ""
set title "dy"
#set yrange [0.00 : 0.0025]
#set ytic 0.00, 0.001, 0.0025
plot @refnew using 1:(($3-$7)) @diff 

set title "dz"
#set yrange [0.00 : 0.0004]
#set ytic 0.00, 0.0002, 0.0004
plot @refnew using 1:(($4-$8)) @diff
