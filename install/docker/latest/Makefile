IMAGE=finmag/finmag:latest

build: Dockerfile Makefile
	docker build -t $(IMAGE) .

test:
	docker run -ti -w="/finmag" --rm $(IMAGE) bash -c "py.test"

run:
	docker run -ti -v $(pwd):/io --rm $(IMAGE)

push:
	docker push $(IMAGE)

pull:
	docker pull $(IMAGE)
