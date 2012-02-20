meshgen:
	netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode

meshconvert:
	nmeshimport --netgen bar.neutral bar.nmesh.h5
	gzip -f bar.neutral

meshshow:
	nmeshpp --vtk bar.nmesh.h5 bar.vtk
	mayavi2 -d bar.vtk -m Surface

run:
	nsim run_nmag.py --clean

extract_M:
	ncol bar time M_Py_0 M_Py_1 M_Py_2 > data_M.txt

extract_H_ex:
	ncol bar time H_exch_Py_0 H_exch_Py_1 H_exch_Py_2 > data_H_ex.txt


