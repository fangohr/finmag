# install prerequisites
sudo apt-get install tk-dev tcl-dev

# download and extract oommf
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
