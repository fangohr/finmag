all: mesh nmag extract 

mesh:
	if test -s film.nmesh; \
	then \
		echo Mesh already exists.; \
	else \
		netgen -geofile=film.geo -meshfiletype="Neutral Format" -meshfile=film.neutral -batchmode; \
		nmeshimport --netgen film.neutral film.nmesh; \
		rm film.neutral; \
	fi

nmag:
	nsim run_nmag.py --clean
	ncol spinwaves time m_Py_0 m_Py_1 m_Py_2 > averages_ref.txt
	
extract:
	nmagpp --vtk=fields spinwaves
