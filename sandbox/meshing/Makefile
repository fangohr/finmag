all: mesh

mesh:
	- netgen -geofile=bar30_30_100.geo -meshfiletype="DIFFPACK Format" -meshfile=bar30_30_100.grid -batchmode
	dolfin-convert bar30_30_100.grid bar30_30_100.xml
	gzip -f bar30_30_100.xml
