build:
	docker build -t finmag .

run: build
	docker run -ti finmag


test:
	cd ../../.. && docker run -ti -v `pwd`:/io/finmag finmag bash -c "cd finmag && make test-fast"
