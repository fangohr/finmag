all: run-nmag run-dolfin

mesh-nmag:
	- netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode
	nmeshimport --netgen bar.neutral bar.nmesh.h5
	gzip -f bar.neutral

run-nmag: mesh-nmag
	nsim run_nmag.py --clean
	ncol bar time M_Py_0 M_Py_1 M_Py_2 > data_M.txt

mesh-dolfin:
	- netgen -geofile=bar.geo -meshfiletype="Gmsh2 Format" -meshfile=bar.gmsh -batchmode
	dolfin-convert bar.gmsh bar.xml

run-dolfin: mesh-dolfin
	python run_dolfin.py

meshshow:
	nmeshpp --vtk bar.nmesh.h5 bar.vtk
	mayavi2 -d bar.vtk -m Surface

