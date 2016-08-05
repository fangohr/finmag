FINMAG_DIR = $(dir $(shell pwd))
build:
	docker build -t finmag .

run:
	docker run -ti -v $(FINMAG_DIR):/io finmag
