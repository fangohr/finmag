# Install fenics
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends fenics
sudo apt-get dist-upgrade -y

# Other dependencies
sudo apt-get install -y git emacs python-pip gmsh netgen cython libsundials-serial-dev \
     libboost-python-dev libboost-test-dev ipython python-scipy python-matplotlib

sudo pip install --upgrade --user pip aeon future sh pytest \
     git+https://github.com/fangohr/dolfinh5tools.git sphinx jupyter nbconvert

# Install netgen.
CURRENT_DIR=$PWD
INSTALL_DIR=netgen_install
mkdir -p netgen_install
sudo apt-get install autoconf tk8.5-dev tcl8.5-dev libxmu-dev gawk python3-tk python3-pip metis openmpi-bin libopenmpi-dev libtogl-dev python3-dev
wget http://kent.dl.sourceforge.net/project/netgen-mesher/netgen-mesher/5.3/netgen-5.3.1.tar.gz 
tar -xvf netgen-5.3.1.tar.gz
cd netgen-5.3.1
autoconf
./configure --with-sysroot=/usr/lib/ --with-tcl=/usr/lib/tcl8.5/ --with-tk=/usr/lib/tk8.5 --with-togl=/usr/lib/ --with-metis=/usr/lib/x86_64-linux-gnu/ --enable-nglib
make 
make install

# Due to a bug in Netgen, we have to disable it's OpenGL calls for the GUI to work. 
# If we dont want the GUI this doesnt really matter.

sudo patch /opt/netgen/bin/drawing.tcl < $CURRENT_DIR/drawing-tcl-patch
echo '============================================================='
echo 'You need to do the following:'
echo ' 1) add /opt/netgen/bin to your path with the command'
echo '      export PATH=/opt/netgen/bin:$PATH'
echo ' 2) Set the environment variable NETGENDIR to /opt/netgen/bin'
echo '      export NETGENDIR=/opt/netgen/bin'
echo ' 3) Add PYTHONPATH to finmag/src'
echo '      export PYTHONPATH=...'
