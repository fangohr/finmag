mesh: mesh

mesh:
	- netgen -geofile=film.geo -meshfiletype="Neutral Format" -meshfile=film.neutral -batchmode
	nmeshimport --netgen film.neutral film.nmesh
	rm film.neutral

nmag:
	nsim run_nmag.py --clean

extract:
	nmagpp --vtk=fields spinwaves
