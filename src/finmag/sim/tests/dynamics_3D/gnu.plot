set multiplot layout 2,1 title "Dynamics 3D"
set title "simulated with nmag"
set xlabel "time (s)"
set ylabel "magnetisation (A/m)"
plot \
    "data_M.txt" using 1:2 title "M_x" with lines, \
    "data_M.txt" using 1:3 title "M_y" with lines, \
    "data_M.txt" using 1:4 title "M_z" with lines
set title "simulated with dolfin"
plot \
    "data_M_dolfin.txt" using 1:2 title "M_x" with lines, \
    "data_M_dolfin.txt" using 1:3 title "M_y" with lines, \
    "data_M_dolfin.txt" using 1:4 title "M_z" with lines
unset multiplot 
