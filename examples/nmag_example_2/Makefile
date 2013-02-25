plots: agreement_averages.png

agreement_averages.png: averages_ref.txt averages.txt
	gnuplot plot.gnu

clean:
	rm -f averages_ref.txt averages.txt finmag_bar.log run_nmag_log.log bar_mat.xml bar.xml.gz agreement_averages.png

.PHONY: plot clean

.INTERMEDIATE: nmag_bar_dat.ndt finmag_bar.ndt bar.nmesh.h5

#
# Rules to create nmag's data.
#

averages_ref.txt: nmag_bar_dat.ndt
	ncol nmag_bar time m_Py_0 m_Py_1 m_Py_2 > averages_ref.txt

nmag_bar_dat.ndt: bar.nmesh.h5
	nsim run_nmag.py --clean

%.nmesh.h5: %.neutral
	nmeshimport --netgen $< $@

%.neutral: %.geo
	- netgen -geofile=$< -meshfiletype="Neutral Format" -meshfile=$@ -batchmode

#
# Rules to create finmag's results.
#

averages.txt: finmag_bar.ndt
	ncol finmag_bar.ndt time m_x m_y m_z > averages.txt

finmag_bar.ndt:
	python run_finmag.py
