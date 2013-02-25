demagnetisationfield.pdf: demagnetisationfield.tex magstyle.sty
	pdflatex demagnetisationfield.tex

force:
	pdflatex demagnetisationfield.tex	

%.neutral: %.geo
	@echo "\n--- DEBUG: Will run netgen ---\n"
	- netgen -geofile=$< -meshfiletype="Neutral Format" -meshfile=$@ -batchmode
	@echo "\n--- DEBUG: Have run netgen, will tell you file size ---\n"
	du $@

%.nmesh.h5: %.neutral
	@echo "\n--- DEBUG: Will run nmeshimport ---\n"
	echo $PATH
	echo `which nmeshimport`
	du $<
	nmeshimport --netgen $< $@
	/var/lib/jenkins/nmag-0.2.1/bin/nmeshimport --netgen $< $@
	@echo "\n--- DEBUG: Have run nmeshimport, will tell you file size ---\n"
	du $@

nmag_data.dat: $(MESH)
	@echo "\n--- DEBUG: Will run nmag ---\n"
	nsim run_nmag.py --clean $(MESH) nmag_data.dat
	@echo "\n--- DEBUG: Have run nmag ---\n"
