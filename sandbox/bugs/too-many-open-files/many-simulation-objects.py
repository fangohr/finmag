import os
import psutil    # sudo apt-get install python-psutils
import dolfin as df
import finmag

# get handle to this process
p = psutil.Process(os.getpid())

def openfiles():
    return p.get_open_files()

def openfilescount():
    return len(openfiles())

def create_finmag_sim_object(name):
    mesh = df.UnitIntervalMesh(1)
    sim = finmag.Simulation(mesh, Ms=1, unit_length=1e-9, name=name)
    return sim

def create_sims(base='sim', start=0, stop=20):
    sims = []

    for i in range(start, stop):
        name = '%s-%04d' % (base, i)
        print("Creating object %s" % name)
        sims.append(create_finmag_sim_object(name))
        print("name=%s, i=%d, %d files open" % (name, i, openfilescount()))
    return sims

sims = create_sims()
sims = sims + create_sims(base='sim2')





