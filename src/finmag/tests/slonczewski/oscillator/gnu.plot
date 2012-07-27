
plot [0:6e-9] \
    "./averages.txt" using 1:2 title "mx finmag" w d lc rgb "blue", \
    "./averages.txt" using 1:3 title "my" w d lc rgb "red", \
    "./averages.txt" using 1:4 title "mz" w d lc rgb "green", \
    "./averages_nmag5.txt" using 1:2 title "mx nmag5" w l lc rgb "blue", \
    "./averages_nmag5.txt" using 1:3 title "my" w l lc rgb "red", \
    "./averages_nmag5.txt" using 1:4 title "mz" w l lc rgb "green"

pause -1
