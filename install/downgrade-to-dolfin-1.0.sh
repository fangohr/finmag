#sudo ppa-purge ppa:fenics-packages/fenics


sudo add-apt-repository ppa:fenics-packages/fenics-1.0.x 
sudo apt-get update
sudo apt-get upgrade

sudo apt-get install libdolfin1.0=1.0.0-2~ppa2~precise1
sudo apt-get install libdolfin1.0-dev=1.0.0-2~ppa2~precise1
sudo apt-get install dolfin-bin=1.0.0-2~ppa2~precise1
sudo apt-get install python-dolfin=1.0.0-2~ppa2~precise1
