all: mesh nmag

mesh:
	- netgen -geofile=coarse_bar.geo -meshfiletype="Neutral Format" -meshfile=coarse_bar.neutral -batchmode
	nmeshimport --netgen coarse_bar.neutral coarse_bar.nmesh.h5
	- netgen -geofile=coarse_bar.geo -meshfiletype="DIFFPACK Format" -meshfile=coarse_bar.grid -batchmode
	dolfin-convert coarse_bar.grid coarse_bar.xml
	gzip -f coarse_bar.xml
	#rm coarse_bar.neutral coarse_bar.grid coarse_bar_bi.xml coarse_bar_mat.xml coarse_bar.xml.bak
	- netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode
	nmeshimport --netgen bar.neutral bar.nmesh.h5
	- netgen -geofile=bar.geo -meshfiletype="DIFFPACK Format" -meshfile=bar.grid -batchmode
	dolfin-convert bar.grid bar.xml
	gzip -f bar.xml
	#rm bar.neutral bar.grid bar_bi.xml bar_mat.xml bar.xml.bak
	
nmag: mesh
	nsim run_nmag.py --clean

finmag: 
	python run_finmag.py
