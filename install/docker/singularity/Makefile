IMAGE=ryanpepper/finmag:singularity

build-docker: Dockerfile Makefile
	docker build -t $(IMAGE) .

pull:
	docker pull $(IMAGE)

push:
	docker push $(IMAGE)

run:
	docker run -ti -v $$(pwd):/io --rm $(IMAGE)

build-singularity: build-docker
	singularity image.create --size 1124 --force finmag.img
	sudo singularity build finmag.img Singularity
