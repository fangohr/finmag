all: nmag

bar_5_5_5.nmesh.h5:
	- netgen -geofile=bar_5_5_5.geo -meshfiletype="Neutral Format" -meshfile=bar_5_5_5.neutral -batchmode
	nmeshimport --netgen bar_5_5_5.neutral bar_5_5_5.nmesh.h5
	rm bar_5_5_5.neutral

clean:
	rm bar_5_5_5.nmesh.h5

nmag: bar_5_5_5.nmesh.h5
	nsim run_nmag.py --clean

meshshow: mesh
	nmeshpp --vtk bar_5_5_5.nmesh.h5 bar_5_5_5.vtk
	mayavi2 -d bar_5_5_5.vtk -m Surface

