#Given a container id, run this container with interactive bash
docker run --privileged -i -t --volume=`pwd`:/home/root:rw  -t $1 bash -i

