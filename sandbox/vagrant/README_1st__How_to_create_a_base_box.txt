NOTE: Before proceeding make sure that you have the latest version of
      VirtualBox installed (at the time of this writing this is 4.3.8).
      You can get it from: https://www.virtualbox.org/wiki/Downloads
      
      Vagrant version 1.5.1 is current at time of writing and can be downloaded
      from http://www.vagrantup.com/downloads.html     

The way Vagrant works is that it uses a so-called 'base box' [1] to
set up a minimal working operating system and then (optionally) runs
certain 'provisioning' instructions (e.g. shell scripts, chef recipes,
etc.) to put the virtual machine into a precise user-defined state.
There are base boxes for the most common operating systems available,
but sometimes they don't quite fit our purpose (e.g. they come without
VirtualBox Guest Additions installed), and I couldn't find one for
Ubuntu 13.10. Also, it's nice to know how to set up a base box manually
anyway. We will do this using a tool called 'packer' in combination with
a set of box definition files from the repository at [3].

Here are the steps:

1) Install the tool 'packer' [2] by running

       ./install_packer.sh

2) Download 'bento' [3] by running the following command:

       git clone https://github.com/opscode/bento.git

3) Change into the subdirectory './bento/packer' of the newly cloned git
   repository:

       cd bento/packer

4) Run the following command:

       packer build -only=virtualbox-iso ubuntu-13.10-amd64.json

   This will install Ubuntu 13.10 suitable for a 64-bit architecture.
   If you want a different version or architecture, replace the name
   of the .json file with the correct name (see the files in the
   ./bento/packer subdirectory for available options).


If successful, these steps will build a 'bare box' which can be used
with Vagrant. Bento puts the box file in the location
 
    ./bento/builds/virtualbox/opscode_ubuntu-13.10_chef-provisionerless.box

but it can be moved anywhere. I order to work use this box with the vagrant
scripts in this directory, move it into the directory ./ubuntu_12.04/downloads/
as follows:

    mv ./bento/builds/virtualbox/opscode_ubuntu-13.10_chef-provisionerless.box ubuntu_12.04/downloads

Then proceed with the steps in the file 'README_2nd'.

[1] http://docs.vagrantup.com/v2/boxes/base.html
[2] http://www.packer.io/
[3] https://github.com/opscode/bento
