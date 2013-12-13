package: instructions.html
	rm -rf finmag-installation-instructions
	mkdir -p finmag-installation-instructions
	echo "Archive created `date`" > finmag-installation-instructions/version.txt
	hg tip | head -n 4 >> finmag-installation-instructions/version.txt
	cp -v instructions.txt instructions.html install-libraries.sh test_libraries.py finmag-installation-instructions
	tar cfvz finmag-installation-instructions.tar.gz finmag-installation-instructions
	rm -rf finmag-installation-instructions
	echo "Tar ball finmag-installation-instructions.tar.gz has been created:"
	ls -lh finmag-installation-instructions.tar.gz

instructions.html: instructions.txt
	rst2html.py instructions.txt instructions.html	