#!/bin/bash

set -o errexit
bold=`tput bold`
normal=`tput sgr0`
cwd=$(pwd)

echo "This script will help you download and compile fenics. Get usage info with -h."

#############################################
# Check if the user wants to skip any steps #
#############################################

step=0
only_update=false
while getopts "uhs:" opt; do
  case $opt in
  s)
      step=$OPTARG
      ;;
  u)
      only_update=true
      ;;
  h)
      echo -e "\n${bold}Usage:${normal}"
      echo "no arguments - install dorsal and FEniCS"
      echo "-s N to jump to a specific step in the installation"
      echo "-u only update your installation of FEniCS"
      exit 0
      ;;
  \?)
      exit 1
      ;;
  :)
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

######################
# Only update Fenics #
######################

if $only_update; then
    source $HOME/.finmag/FENICS_PREFIX.cfg  # this was created during the installation
    cd $DORSAL_PREFIX/dorsal
    ./dorsal.sh raring.platform  # this has dependencies marked with skip
    exit 0
fi

###########################
# Step 1. Download Dorsal #
###########################

DORSAL_PREFIX=${DORSAL_PREFIX:-$HOME} # will download dorsal to $HOME by default

if [ ${step} -le 1 ];
then
    echo -e "\n${bold}(step 1/5)${normal} Dorsal will be downloaded to '$DORSAL_PREFIX'."
    echo "The directory can be changed by setting the environment variable DORSAL_PREFIX."
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if ! [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting. Please set DORSAL_PREFIX to the desired download directory and try again."
        exit 0
    fi
    git clone git@bitbucket.org:fenics-project/dorsal.git $DORSAL_PREFIX/dorsal
else
    echo -e "\n${bold}(step 1/5)${normal} Skipping download of Dorsal."
fi

############################
# Step 2. Configure Dorsal #
############################

FENICS_PREFIX=${FENICS_PREFIX:-$HOME} # will download fenics to $HOME by default

if [ ${step} -le 2 ];
then
    echo -e "\n${bold}(step 2/5)${normal} Configuring Dorsal."
    echo "Please confirm installation of fenics into '$FENICS_PREFIX'."
    echo "This can be changed by setting the environment variable FENICS_PREFIX."
    read -p "Continue? (y/n) " -n 1 -r
    echo
    if ! [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborting. Please set FENICS_PREFIX to the desired download directory and try again."
        exit 0
    fi
    cp local.cfg $DORSAL_PREFIX/dorsal  # copy our custom dorsal config
    cp raring.platform $DORSAL_PREFIX/dorsal  # copy our custom platform file
else
    echo -e "\n${bold}(step 2/5)${normal} Skipping configuration of Dorsal."
fi

######################
# Step 3. Run Dorsal #
######################

if [ ${step} -le 3 ];
then
    echo -e "\n${bold}(step 3/5)${normal} Running Dorsal."
    export FENICS_PREFIX  # dorsal will look this up when it reads our local.cfg
    cd $DORSAL_PREFIX/dorsal
    ./dorsal.sh
    cd $cwd
else
    echo -e "\n${bold}(step 3/5)${normal} Skipping running Dorsal at all."
fi

############################################################
# Step 4. Save installation directories for future updates #
############################################################

if [ ${step} -le 4 ];
then
    echo -e "\n${bold}(step 4/5)${normal} Remembering installation directories for future convenience."
    if [ ! -d $HOME"/.finmag" ]; then
        mkdir $HOME"/.finmag"
    fi
    echo -e "export DORSAL_PREFIX="$DORSAL_PREFIX\
        "\nexport FENICS_PREFIX="$FENICS_PREFIX > $HOME/.finmag/FENICS_PREFIX.cfg  # remember for updates
else
    echo -e "\n${bold}(step 4/5)${normal} Skipping remember of installation directories." 
fi

#############################################################################
# Step 5. Copy platform file to dorsal that will skip building dependencies #
#############################################################################

if [ ${step} -le 5 ];
then
    echo -e "\n${bold}(step 5/5)${normal} Copying platform file for future updates of FEniCs."
    cp raring.platform $DORSAL_PREFIX/dorsal  # copy our custom platform file
else
    echo -e "\n${bold}(step 5/5)${normal} Skipping copying of platform file for future updates." 
fi
