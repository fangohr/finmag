from dolfin import tic, toc
from test_exchange_demag import run_finmag
from finmag.util.timings import timings
import commands

def run_nmag():
    cmd = "nsim run_nmag.py --clean"
    status, output = commands.getstatusoutput(cmd)
    return status

tic()
status = 0#run_nmag()
T1 = toc()
if status != 0:
    import sys
    sys.exit(1)

tic()
run_finmag()
T2 = toc()

print timings
print "Nmag time: %.4f sec" % T1
print "Finmag time: %.4f sec" % T2
