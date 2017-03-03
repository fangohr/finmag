run: finmag-compiled.done
	@echo "Expect to have a finmag repository checked out in ./finmag ."
	docker run -ti -v `pwd`:/io buildbot bash


# build docker container if not done yet
docker-image.done: Dockerfile
	docker build -t buildbot .
	date > docker-image.done



# clone firmag if note done yet
finmag-repo-clone.done: docker-image.done 
	# clone source code on host
	git clone git@github.com:fangohr/finmag.git
	# select right branch
	cd finmag && git checkout master
	touch finmag-repo-clone.done

finmag-compiled.done: finmag-repo-clone.done

	docker run -ti -v `pwd`:/io buildbot python finmag/install/memory-warning.py
	# this line should come later
	# but at the moment, the import fails,
	# so we are always in this target
	date > finmag-compiled.done

	# import finmag in container to build binary files
	# (only makes sense if changes are persistent -> need to move location
	#  of temporary files from /tmp to somewhere else. But where?)
	# (de-activated for now)
	#docker run -ti -v `pwd`:/io buildbot python -c "import finmag"


clean:
	rm -vf finmag-setup.done docker-image.done finmag-compiled.done
	@echo "If you have no changes in finmag, please remove the directory"
	@echo "i.e. rm -rf ./finmag"
