plot \
    "./averages_nmag5_no_stt.txt" using 1:2 title "mx nmag5" with lines lc rgbcolor "blue", \
    "./averages_nmag5_no_stt.txt" using 1:3 title "my" with lines lc rgbcolor "red", \
    "./averages_nmag5_no_stt.txt" using 1:4 title "mz" with lines lc rgbcolor "green", \
    "./averages_no_stt.txt" using 1:2 title "mx finmag" with points lc rgbcolor "blue" linewidth 2, \
    "./averages_no_stt.txt" using 1:3 title "my" with points lc rgbcolor "red" linewidth 2, \
    "./averages_no_stt.txt" using 1:4 title "mz" with points lc rgbcolor "green" linewidth 2

pause -1
