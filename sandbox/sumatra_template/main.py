import os
import sys
from sumatra.parameters import build_parameters

# The following line is important because Sumatra creates a parameter file
# 'on the fly' and passes its name to the script, so we should *not* use a
# hard-coded filename here.
paramsfile = sys.argv[1]
parameters = build_parameters(paramsfile)

# I like printing the sumatra label of this run:
smt_label = parameters['sumatra_label']
print "Sumatra label for this run: {}".format(smt_label)
sys.stdout.flush()

# Change into the datastore directory to run the simulation there.
# Note that this has to happen *after* reading the parameter above,
# otherwise it won't find the parameter file.
os.chdir(os.path.join('Data', smt_label))


# The variable 'parameters' defined above is a dictionary associating
# each parameter name with its value, so we can use this neat trick to
# make the parameters available as global variables:
globals().update(parameters)

# Alternatively, if we don't want to resort to "black magic", we can
# assign each parameter value separately to a variable:
Msat = parameters['Msat']
H_ext = parameters['H_ext']
A = parameters['A']
# ... etc. ...

#
# The main part of the script follows here.
#
