all: create run

# Create does two things:
# 1) Creates the container filesystem in the file container.img
# 2) Sets up the container.img file using the config file Singularity.

create:
	sudo singularity create --size 8192 container.img
	sudo singularity bootstrap container.img Singularity

# Clean removes the container image
clean:
	rm -rf container.img

# Run allows you to run the container only!
run:
	singularity shell container.img


# Edit runs the container, but also allows you to edit it, e.g. to install a new package.
edit:
	sudo singularity shell --writable container.img
