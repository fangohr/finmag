FINMAG_DIR = $(dir $(shell pwd))
build:
	docker build -t finmag/ubuntu_1604_base .

push: build
	docker push finmag/ubuntu_1604_base

run:
	docker run -ti -v $(FINMAG_DIR):/io finmag/ubuntu_1604_base
