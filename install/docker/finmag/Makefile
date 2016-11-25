build:
	docker build -t buildbot .

run:
	@echo "You should have a finmag repository checked out in ./finmag ."
	docker run -ti -v `pwd`:/io buildbot bash
