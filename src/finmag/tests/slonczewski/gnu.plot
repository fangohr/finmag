
plot [0:6e-9] \
    "./m_averages_validate.txt" using 1:2 title "mx finmag" lc rgb "blue", \
    "./m_averages_validate.txt" using 1:3 title "my" lc rgb "red", \
    "./m_averages_validate.txt" using 1:4 title "mz" lc rgb "green", \
    "./averages_ref.txt" using 1:2 title "mx nmag5" w l lc rgb "blue", \
    "./averages_ref.txt" using 1:3 title "my" w l lc rgb "red", \
    "./averages_ref.txt" using 1:4 title "mz" w l lc rgb "green"

pause -1
