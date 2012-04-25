all: nmag

mesh:
	- netgen -geofile=bar_5_5_5.geo -meshfiletype="Neutral Format" -meshfile=bar_5_5_5.neutral -batchmode
	nmeshimport --netgen bar_5_5_5.neutral bar_5_5_5.nmesh.h5
	gzip -f bar_5_5_5.neutral

nmag:
	nsim run_nmag.py --clean

meshshow: mesh
	nmeshpp --vtk bar_5_5_5.nmesh.h5 bar_5_5_5.vtk
	mayavi2 -d bar_5_5_5.vtk -m Surface

