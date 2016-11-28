import sys
from psutil import virtual_memory

mem = virtual_memory()
mem.total  # total physical virtual_memory

GB = float(1024**3)   # float need if we run with python 2

if mem.total / GB < 3.8:
    print("Warning: building native modules may fail due to not enough physical memory.")
    print("You have {:.1f} GB available.\n".format(mem.total/GB))
    print("\tContext: The C++ compiler needs lots of RAM. 4GB seem sufficient (Nov 2016)\n")
    print("\tIf you are on OSX / Windows, and using Docker, try this")
    print("""\t\tdocker-machine stop
\t\tVBoxManage modifyvm default --memory 4096
\t\tdocker-machine start\n""")
    print("\tSource: http://stackoverflow.com/questions/32834082/how-to-increase-docker-machine-memory-mac")
    sys.exit(1)
