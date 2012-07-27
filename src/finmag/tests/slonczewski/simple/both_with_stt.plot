plot \
    "./averages.txt" using 1:2 title "mx finmag (STT)" with points lc rgbcolor "blue", \
    "./averages.txt" using 1:3 title "my" with points lc rgbcolor "red", \
    "./averages.txt" using 1:4 title "mz" with points lc rgbcolor "green", \
    "./averages_nmag5.txt" using 1:2 title "mx nmag5 (STT)" with lines lc rgbcolor "blue" linewidth 2, \
    "./averages_nmag5.txt" using 1:3 title "my" with lines lc rgbcolor "red" linewidth 2, \
    "./averages_nmag5.txt" using 1:4 title "mz" with lines lc rgbcolor "green" linewidth 2

pause -1
