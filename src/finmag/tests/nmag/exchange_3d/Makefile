all: nmag

mesh:
	- netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode
	nmeshimport --netgen bar.neutral bar.nmesh.h5
	gzip -f bar.neutral

nmag: mesh
	nsim run_nmag.py --clean
	ncol bar time m_Py_0 m_Py_1 m_Py_2 > averages_ref.txt

meshshow:
	nmeshpp --vtk bar.nmesh.h5 bar.vtk
	mayavi2 -d bar.vtk -m Surface

