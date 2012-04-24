all: xmlmesh h5mesh nmag finmag cleanup

xmlmesh:
	- netgen -geofile=bar.geo -meshfiletype="DIFFPACK Format" -meshfile=bar.grid -batchmode
	dolfin-convert bar.grid bar.xml
	gzip -f bar.xml

h5mesh:
	- netgen -geofile=bar.geo -meshfiletype="Neutral Format" -meshfile=bar.neutral -batchmode
	nmeshimport --netgen bar.neutral bar.nmesh.h5


nmag:
	nsim run_nmag.py --clean

finmag:
	python run_finmag.py

cleanup:
	rm bar_bi.xml bar.grid bar_mat.xml bar.neutral bar.xml.bak run_nmag_log.log bar_dat.ndt
	echo Results written to results.rst.
