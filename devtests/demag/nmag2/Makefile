all: nmag

mesh:
	- netgen -geofile=bar30_30_100.geo -meshfiletype="Neutral Format" -meshfile=bar30_30_100.neutral -batchmode
	nmeshimport --netgen bar30_30_100.neutral bar30_30_100.nmesh.h5
	gzip -f bar30_30_100.neutral

nmag: mesh
	nsim run_nmag.py --clean
	ncol bar time m_Py_0 m_Py_1 m_Py_2 > averages_ref.txt

meshshow:
	nmeshpp --vtk bar.nmesh.h5 bar.vtk
	mayavi2 -d bar.vtk -m Surface

