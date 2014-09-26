# This is a comment
FROM fenicsproject/stable-ppa:1.4.0
RUN apt-get update
RUN apt-get -y install gdebi python-pip
ADD ./finmag.deb /tmp/finmag.deb
ADD ./.set-MAC-address.sh /root/.set-MAC-address.sh 
RUN mkdir -p /root/.finmag
ADD ./License /root/.finmag/License
MAINTAINER finmag-team <fangohr@soton.ac.uk>
# install finmag with dependencies
RUN sudo gdebi --non-interactive /tmp/finmag.deb 
RUN sudo pip install aeon sh
# for easier debugging the docker file creation
#RUN sudo apt-get install aptitude
RUN sudo apt-get install -y libboost-python-dev python-netifaces



# the right command to run this docker file for interactive exploration: 
# something like: 
# sudo docker run -i -t 52762ca8da0a bash -i
# where the 52762ca8da0a needs to be replaced by the right key that is displayed
# at the end of the successful dockerfile creation.

#
# sudo docker run -i -t --volume=`pwd`:/home/root:rw  -t 9ad62e6d390a bash -i 
