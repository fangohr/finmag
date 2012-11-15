all: nmag

nmag: energies_ref.txt averages_ref.txt

averages_ref.txt: bar_dat.ndt
	ncol bar time m_Py_0 m_Py_1 m_Py_2 > averages_ref.txt

energies_ref.txt: bar_dat.ndt
	ncol bar E_exch_Py E_demag_Py > energies_ref.txt

bar_dat.ndt: bar30_30_100.nmesh.h5
	nsim run_nmag.py --clean

bar30_30_100.nmesh.h5:
	- netgen -geofile=bar30_30_100.geo -meshfiletype="Neutral Format" -meshfile=bar30_30_100.neutral -batchmode
	nmeshimport --netgen bar30_30_100.neutral bar30_30_100.nmesh.h5
	gzip -f bar30_30_100.neutral

meshshow:
	nmeshpp --vtk bar.nmesh.h5 bar.vtk
	mayavi2 -d bar.vtk -m Surface

finmag:
	python test_exchange_demag.py

show:
	gnuplot gnu.plot
