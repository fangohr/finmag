IMAGE=finmag/finmag:dependencies

build: Dockerfile Makefile
	docker build -t $(IMAGE) .

pull:
	docker pull $(IMAGE)

push:
	docker push $(IMAGE)

run:
	docker run -ti -v $$(pwd):/io --rm $(IMAGE)
