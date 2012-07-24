plot \
    "./averages_no_stt.txt" using 1:2 title "mx finmag" with dots lc rgbcolor "blue", \
    "./averages_no_stt.txt" using 1:3 title "my" with dots lc rgbcolor "red", \
    "./averages_no_stt.txt" using 1:4 title "mz" with dots lc rgbcolor "green", \
    "./averages.txt" using 1:2 title "mx finmag (STT)" with lines lc rgbcolor "blue" linewidth 2, \
    "./averages.txt" using 1:3 title "my" with lines lc rgbcolor "red" linewidth 2, \
    "./averages.txt" using 1:4 title "mz" with lines lc rgbcolor "green" linewidth 2

pause -1
