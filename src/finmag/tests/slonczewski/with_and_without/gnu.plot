plot \
    "./m_averages_without.txt" using 1:2 title "mx" with lines lc rgbcolor "blue", \
    "./m_averages_without.txt" using 1:3 title "my" with lines lc rgbcolor "red", \
    "./m_averages_without.txt" using 1:4 title "mz" with lines lc rgbcolor "green", \
    "./m_averages_with.txt" using 1:2 title "mx (STT)" with lines lc rgbcolor "blue" linewidth 2, \
    "./m_averages_with.txt" using 1:3 title "my" with lines lc rgbcolor "red" linewidth 2, \
    "./m_averages_with.txt" using 1:4 title "mz" with lines lc rgbcolor "green" linewidth 2

pause -1
