Preparatory steps
=================

1) Getting a base box

   In order to use vagrant you will need a 'base box' which provides
   a bare-bones Ubuntu installation. You can follow the steps in the
   file 'README_1st' to build one from scratch. This is the recommended
   way (it may take a little while but you only need to do it once).

   Alternatively, you can use one of the pre-packaged boxes available
   on the web (e.g. at http://www.vagrantbox.es/). If you specify the
   URL of the .box file in the file 'ubuntu_12.04/Vagrantfile' (see
   configuration option 'config.vm.box_url') then the .box image will
   be downloaded automatically. However, be aware that most of these
   box images are not 100% suited for our purpose (e.g. they might not
   have the correct version of the VirtualBox Guest Additions installed).
   Therefore it is highly recommended to build your own base box as
   described in the README_1st file.

2) SSH keys for passwordless access to Bitbucket

   In order to clone the finmag and finmag-dist repositories, vagrant
   needs passwordless access to Bitbucket. In order to enable this,
   place suitable SSH keys (e.g. the ones from Marvin on aleph0, which
   are called 'jenkins_id_rsa' and 'jenkins_id_rsa.pub') into the folder
   './ubuntu_12.04/shared_folder/'


Quick getting-started guide
===========================

After building or obtaining a base box (which should be put in the
directory ubuntu_12.04/downloads), all you need to do is:

   cd ubuntu_12.40 && vagrant up

The 'vagrant up' command command will import the base box (potentially
after downloading it, in case you specified a base box URL from the web)
and install Finmag along with all necessary prerequisites (including
FEniCS, OOMMF, Magpar and Nmag).

Note that by default, Vagrant will store the imported box image in the
subfolder ~/.vagrant.d of your home directory. If you want them in a
different place (e.g. because disk space is limited), set the
VAGRANT_HOME environment variable to the desired location. For
example, you can put this in your ~/.bashrc file:

   export VAGRANT_HOME=/path/to/desired/location/


Background information
======================

This directory contains a first attempt to set up Vagrant [1] for use
with Finmag. The purpose of Vagrant is to install a virtual machine
(in our case a VirtualBox image, but Vagrant supports others as well)
in a precisely defined state, for example with additional software
installed. For our purposes this is useful for two things:

  - In order to build a binary distribution of Finmag, ideally we need
    to compile the code on the same operating system as the one that
    the Finmag binary is going to run on. With a virtual machine, this
    can be easily done and even automised so that it could be part of
    the regular Jenkins test runs.

  - We use a bash script to set up the virtual machine in the
    pre-configured state that is necessary to run Finmag on it. Among
    other things, this script installs the FEniCS PPA, installs the
    necessary dependencies for Finmag, installs OOMMF/Magpar/Nmag,
    etc. This script is also useful for people who already have an
    Ubuntu installation (e.g. independently of a virtual machine) and
    just want to install all the necessary prerequisites to run Finmag
    on their system. By having this script as part of the Vagrant
    setup and hopefully integrated into Jenkins we could be sure that
    it is always up-to-date (and tested).


Vagrant uses a so-called Vagrantfile to choose the right 'base box'
for the virtual machine which it should set up. This Vagrantfile
contains some configuration items about the desired operating system
(e.g. Ubuntu 12.04 or 13.10), how much memory the virtual box should
use, etc. Currently we keep the Vagrantfiles for different operating
systems in separate subfolders to keep things tidy (in fact, at the
moment we only have one for Ubuntu 13.10), but there is probably a
command-line switch or some other configuaration option which which
would allow us to merge them.

A list of available base boxes for different operating systems is
available at [2]. Note that the base box is downloaded from the
correct location each time you say 'vagrant up' to set up a new
virtual machine from scratch. To avoid this, you can download the file
once by saying 'make download-base-box' in the directory which
contains the Vagrantfile.

After installing the 'bare box' (which only installs a base Ubuntu
operating system, for example), Vagrant will run the script
'provision.sh'. This is just a shell script which contains the
necessary commands to install all Finmag prerequisites as well as
Finmag itself and to run the tests.


[1] http://www.vagrantup.com/
[2] http://www.vagrantbox.es/
