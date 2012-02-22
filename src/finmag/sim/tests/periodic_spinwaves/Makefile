mesh: meshgen meshconvert meshmirror

meshgen:
	- netgen -geofile=film.geo -meshfiletype="Neutral Format" -meshfile=film.neutral -batchmode

meshconvert:
	nmeshimport --netgen film.neutral film.nmesh
	rm film.neutral

meshmirror:
	nmeshmirror film.nmesh 1e-6 1e-6 -1,0,0 periodic.nmesh
	rm film.nmesh
