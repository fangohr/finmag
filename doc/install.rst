Installation
------------

Finmag may be installed on a fresh installation of Ubuntu 12.04 as follows (and the procedure for other Linux distributions will be similar).

1. At a shell prompt, navigate to the directory in which you would like the ``finmag`` tree located.
2. ``hg clone https://nsob104@bitbucket.org/fangohr/finmag``
3. ``cd finmag/install/on-fresh-ubuntu-11.10``
4. Run ``sh do-all.sh``.  At the time of writing, this script requires some interaction.  It invokes ``aptitude`` and ``apt-get`` through ``sudo`` several times, so you will need to be ready to provide the administrative password.  Moreover, some of the package management operations will ask for confirmation before installing packages and before adding the FEniCS PPA to the ``/etc/apt/sources.list.d/`` directory.

At this point, finmag should be installed (should any of the above scripts have failed, check your Internet connectivity before re-trying the script).  However, in order for the included tests to run, it is necessary to install some additional software.  By default, subdirectories under $HOME are made for each piece of additional software installed; this location may be edited in the individual scripts named below.

5. ``cd ..`` to navigate to the ``finmag/install`` directory.
6. Run ``sh oommf.sh`` and when required, provide the administrative password (this script places a simple oommf launch-script in ``/usr/local/bin/``).
7. The script will print information about setting two environment variables, which you should act upon.  Note that you cannot copy and paste the output from the script into a Bash shell startup file due to the spaces around the ``=`` signs.  Remove these spaces and add to your shell's startup file.  For example, for Bash:
``cat >> ~/.profile << EOF
OOMMF_TCL_INCLUDE_DIR=/usr/include/tcl8.5/
OOMMF_TK_INCLUDE_DIR=/usr/include/tcl8.5/

EOF``
8. Run ``sh magpar.sh``
9. Run ``sh nmag.sh`` (this script will require an administrative password as it installs prerequisites via a ``sudo apt-get`` command).
10. Add the directories for ``nmag`` and ``magpar`` to your PATH, and add Finmag to your PYTHONPATH (replacing ``...`` by the parent directory that you chose for your Finmag installation):
``cat >> ~/.profile << EOF
export PATH="$HOME/nmag-0.2.1/nsim/bin:$HOME/magpar-0.9/src/:$PATH"
export PYTHONPATH=".../finmag/src/"
EOF``

Now, ensure that the environment variables are as set in ``~/.profile`` and then run the test suite to verify the Finmag installation:

11. ``. ~/.profile``
12. ``cd ../src && py.test -v``

Expected output will include many lines of tests, each of which should have the status ``PASSED``, and end with some statistics, for example::

================================================================================= 131 passed in 108.40 seconds ==================================================================================
 DVODE--  At current T (=R1), MXSTEP (=I1) steps   
       taken on this call before reaching TOUT     
      In above message,  I1 =      5000
      In above message,  R1 =  0.8000369437165D+00

