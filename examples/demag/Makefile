demagnetisationfield.pdf: demagnetisationfield.tex magstyle.sty
	pdflatex demagnetisationfield.tex

force:
	pdflatex demagnetisationfield.tex	

%.neutral: %.geo
	@echo "\n--- DEBUG: Will run netgen ---\n"
	- netgen -geofile=$< -meshfiletype="Neutral Format" -meshfile=$@ -batchmode
	@echo "\n--- DEBUG: Have run netgen ---\n"

%.nmesh.h5: %.neutral
	@echo "\n--- DEBUG: Will run nmeshimport ---\n"
	nmeshimport --netgen $< $@
	@echo "\n--- DEBUG: Have run nmeshimport ---\n"

nmag_data.dat: $(MESH)
	@echo "\n--- DEBUG: Will run nmag ---\n"
	nsim run_nmag.py --clean $(MESH) nmag_data.dat
	@echo "\n--- DEBUG: Have run nmag ---\n"
