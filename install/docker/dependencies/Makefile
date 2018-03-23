IMAGE=finmag/finmag:dependencies

build: Dockerfile Makefile
	docker build -t $(IMAGE) .

pull:
	docker pull $(IMAGE)

push: build
	docker push $(IMAGE)

run: build
	docker run -ti -v $$(pwd):/io $(IMAGE)
