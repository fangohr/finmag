all: mesh nmag extract 

mesh:
	if test -s mesh.nmesh; \
	then \
		echo Mesh already exists.; \
	else \
		netgen -geofile=mesh.geo -meshfiletype="Neutral Format" -meshfile=mesh.neutral -batchmode; \
		nmeshimport --netgen mesh.neutral mesh.nmesh; \
		rm mesh.neutral; \
	fi

nmag:
	nsim run_nmag5.py --clean
	ncol run_nmag5 time m_Py_0 m_Py_1 m_Py_2 > averages_nmag5.txt
	
nmag-no-stt:
	nsim run_nmag5_no_stt.py --clean
	ncol nmag_no_stt time m_Py_0 m_Py_1 m_Py_2 > averages_nmag5_no_stt.txt
	
extract:
	nmagpp --vtk=fields run_nmag5
