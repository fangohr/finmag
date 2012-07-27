#version control
sudo apt-get install mercurial
#time integration / c++ 
sudo apt-get install libboost-python-dev libboost-thread-dev libsundials-serial-dev libboost-test-dev
#useful (and mostly required) support libraries
sudo apt-get install python-matplotlib python-visual python-scipy ipython python-py python-progressbar


#for documentation with sphinx
sudo apt-get install texlive-latex-extra texlive-latex-recommended python-pygments texlive-fonts-recommended

#Need more recent version of sphinx than ubuntu version
sudo aptitude remove python-sphinx  python-py
#so we can use easy_install:
sudo apt-get install python-setuptools
sudo easy_install -U sphinx
sudo easy_install -U pytest


