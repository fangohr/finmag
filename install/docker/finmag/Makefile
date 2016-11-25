run: docker-image.done finmag
	@echo "Expect to have a finmag repository checked out in ./finmag ."
	docker run -ti -v `pwd`:/io buildbot bash


# build docker container if not done yet
docker-image.done: Dockerfile
	docker build -t buildbot . && date > docker-image.done

# clone firmag if note done yet
finmag:
	git clone git@github.com:fangohr/finmag.git

