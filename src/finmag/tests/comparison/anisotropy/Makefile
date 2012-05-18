all: nmag

bar.nmesh.h5:
	- netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode
	nmeshimport --netgen bar.neutral bar.nmesh.h5
	rm bar.neutral

clean:
	rm bar.nmesh.h5

nmag: bar.nmesh.h5
	nsim run_nmag.py --clean

meshshow: mesh
	nmeshpp --vtk bar.nmesh.h5 bar.vtk
	mayavi2 -d bar.vtk -m Surface

