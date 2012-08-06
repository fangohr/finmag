#
# This script will download and install oommf into a directory
# called oommf under $PREFIX, which is your user's home per default. It will
# then create a command called "oommf" in your /usr/local/bin directory.
#
# It will also install the needed ubuntu packages.
#

# The default installation location is $HOME. Set
# the PREFIX environment variable to change this.
PREFIX=${PREFIX:-$HOME}

echo "Installing oommf in '$PREFIX'. Set the PREFIX environment variable to specify a different location."


# install prerequisites
sudo apt-get install tk-dev tcl-dev

# create installation directory if it doesn't exist
if ! [ -e ${PREFIX} ]; then
   install -d ${PREFIX};
   echo "Creating directory $PREFIX";
fi

# download and extract oommf
cd $PREFIX
if [ ! -e "oommf12a4pre-20100719bis.tar.gz" ]
then
    wget http://math.nist.gov/oommf/snapshot/oommf12a4pre-20100719bis.tar.gz
fi
tar -xzf oommf12a4pre-20100719bis.tar.gz
mv oommf12a4pre-20100719bis oommf
cd oommf

# install oommf
OOMMF_TCL_INCLUDE_DIR=/usr/include/tcl8.5/; export OOMMF_TCL_INCLUDE_DIR
OOMMF_TK_INCLUDE_DIR=/usr/include/tcl8.5/; export OOMMF_TK_INCLUDE_DIR
./oommf.tcl pimake distclean
./oommf.tcl pimake upgrade
./oommf.tcl pimake
./oommf.tcl +platform

# create an executable called 'oommf' to call oommf in /usr/local/bin
oommf_command=$(cat <<EOF
#! /bin/bash
tclsh $PREFIX/oommf/oommf.tcl "\$@"
EOF
)
sudo sh -c "echo '$oommf_command' > '/usr/local/bin/oommf'"
sudo chmod a+x /usr/local/bin/oommf
