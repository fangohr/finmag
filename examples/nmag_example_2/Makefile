all:
	make averages.txt
	make averages_ref.txt
	echo "Results plotted in comparison.png"

bar.nmesh.h5:
	- netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode
	nmeshimport --netgen bar.neutral bar.nmesh.h5
	rm bar.neutral

bar.xml.gz:
	- netgen -geofile=bar.geo -meshfiletype="DIFFPACK Format" -meshfile=bar.grid -batchmode
	dolfin-convert bar.grid bar.xml
	gzip -f bar.xml
	rm bar.grid bar_marker_*.xml

averages_ref.txt: bar.nmesh.h5
	nsim run_nmag.py --clean
	ncol nmag_bar time m_Py_0 m_Py_1 m_Py_2 > averages_ref.txt

averages.txt: bar.xml.gz
	python run_finmag.py
	ncol finmag_bar.ndt time m_x m_y m_z > averages.txt

plot: averages_ref.txt averages.txt
	gnuplot plot.gnu

clean:
	rm nmag_bar_dat.ndt averages_ref.txt finmag_bar.ndt averages.txt

.PHONY: plot clean
