all: nmag

sphere.nmesh.h5:
	- netgen -geofile=sphere.geo -meshfiletype="Neutral Format" -meshfile=sphere.neutral -batchmode
	nmeshimport --netgen sphere.neutral sphere.nmesh.h5
	rm sphere.neutral

nmag: sphere.nmesh.h5
	nsim run_nmag.py --clean

meshshow: sphere.nmesh.h5
	nmeshpp --vtk sphere.nmesh.h5 sphere.vtk
	mayavi2 -d sphere.vtk -m Surface

