# FinMag - a thin layer on top of FEniCS to enable micromagnetic multi-physics simulations
# Copyright (C) 2012 University of Southampton
# Do not distribute
#
# CONTACT: h.fangohr@soton.ac.uk
#
# AUTHOR(S) OF THIS FILE: Dmitri Chernyshenko (d.chernyshenko@soton.ac.uk)

import os
import re
import hashlib
import tempfile
import numpy as np
import cStringIO
import sys
import subprocess
import shutil
from finmag.util.oommf import ovf, lattice
from finmag.util.oommf.mesh import MeshField, Mesh
from subprocess import check_output, CalledProcessError

CACHE_DIR = os.environ['HOME'] + "/.oommf_calculator"
RUN_DIR = tempfile.mkdtemp(suffix='_oommf_calculator')

if os.environ.has_key('OOMMF_COMMAND'):
    OOMMF_COMMAND=os.environ['OOMMF_COMMAND']
else:
    OOMMF_COMMAND='oommf'

MIF_TEMPLATE="""# MIF 2.1

%(spec)s

Specify Oxs_RungeKuttaEvolve:evolver {
   gamma_G %(gamma_G)s
   alpha %(alpha)s
   method rkf54
}

Specify Oxs_TimeDriver {
    basename %(basename)s
    evolver :evolver
    mesh :mesh
    total_iteration_limit 1
    Ms %(Ms)s
    m0 { Oxs_FileVectorField  {
        atlas :atlas
        norm  1.0
        file %(basename)s-start.omf
    }}
}

Destination archive mmArchive:oommf_calculator
%(fields)s
"""

SOURCE = open(os.path.abspath(__file__)).read()

def run_oommf(dir, args, **kwargs):
    try:
        cmd = [OOMMF_COMMAND]
        cmd.extend(args)
        check_output(cmd, cwd=dir, stderr=subprocess.STDOUT, **kwargs)
    except CalledProcessError, ex:
        sys.stderr.write(ex.output)
        raise Exception("OOMMF invocation failed: " + " ".join(cmd))
    except OSError, ex:
        sys.stderr.write(ex.strerror+".\n")
        raise Exception("Command '{0}' failed. Parameters: '{1}'.".format(cmd[0],  " ".join(cmd[1:])))

# Runs an OOMMF mif file contained in str
# Returns a hashtable of field names mapped to arrays compatible with the given mesh
def calculate_oommf_fields(name, s0, Ms, spec=None, alpha=0., gamma_G=0., fields=[]):
    assert type(Ms) is float
    assert type(s0) is MeshField and s0.dims == (3,)

    ## Calculate the checksum corresponding to the parameters
    m = hashlib.new('md5')
    delim="\n---\n"
    m.update(SOURCE + delim)
    m.update(name + delim)
    m.update("%25.19e%s" % (Ms, delim))
    m.update("%25.19e%s" % (alpha, delim))
    m.update("%25.19e%s" % (gamma_G, delim))
    m.update("%s%s" % (",".join(fields), delim))
    m.update(spec + delim)
    s = cStringIO.StringIO()
    np.save(s, s0.flat)
    m.update(s.getvalue())
    checksum = m.hexdigest()

    ## Format the simulation script
    basename = "%s_%s" % (name, checksum)
    tag = basename.lower()
    params = {
        'spec': spec,
        'basename': basename,
        'Ms': "%25.19e" % Ms,
        'gamma_G': "%25.19e" % gamma_G,
        'alpha': "%25.19e" % alpha,
        'tag': tag,
        'fields': "\n".join("Schedule %s archive Step 1" % f for f in fields)
    }

    mif = MIF_TEMPLATE % params

    #print mif

    # Check if the result is already known
    cachedir = os.path.join(CACHE_DIR, basename)
    try:
        os.makedirs(CACHE_DIR)
    except OSError:
        pass

    if not os.path.exists(cachedir):
        ## Run the simulation
        print "Running OOMMF simulation %s..." % basename,
        sys.stdout.flush()
        dir = os.path.join(RUN_DIR, basename)
        try:
            os.makedirs(dir)
        except OSError:
            pass
        # Write the MIF file
        mif_file_name = basename + ".mif"
        mif_file = open(os.path.join(dir, mif_file_name), "w")
        mif_file.write(mif)
        mif_file.close()
        # Write the starting OMF file
        fl = lattice.FieldLattice(s0.mesh.get_lattice_spec())
        fl.field_data = s0.flat
       
        # Save it to file
        m0_file = ovf.OVFFile()
        m0_file.new(fl, version=ovf.OVF10, data_type="binary8")
        m0_file.write(os.path.join(dir, basename + "-start.omf"))
        # Run the OOMMF simulation
        run_oommf(dir, ["boxsi", "-threads", "4", mif_file_name])
        # Move the results to the cache directory
        shutil.move(dir, cachedir)
        print "success"

    # Read the results
    fields = {}
    for fn in os.listdir(cachedir):
        m = re.match("^(.*)_%s-(.*)-00-0000000.o[hvm]f$" % checksum, fn)
        if m and m.group(1) == name:
            fl = ovf.OVFFile(os.path.join(cachedir, fn)).get_field()
            fields[m.group(2)] = s0.mesh.field_from_xyz_array(fl.field_data)

    return fields

if __name__=="__main__":
    spec = """set pi [expr 4*atan(1.0)]
set mu0 [expr 4*$pi*1e-7]

Parameter cellsize 5e-9

set Hx  -24.6  ;# Applied field in mT
set Hy    4.3
set Hz 0.0

Specify Oxs_BoxAtlas:atlas {
  xrange {0 500e-9}
  yrange {0 125e-9}
  zrange {0   3e-9}
}

Specify Oxs_RectangularMesh:mesh [subst {
  cellsize {$cellsize $cellsize 3e-9}
  atlas Oxs_BoxAtlas:atlas
}]

Specify Oxs_Demag {}
"""

    mesh = Mesh((100, 25, 1), cellsize=(5e-9, 5e-9, 3e-9))
    calculate_oommf_fields("testpppp", mesh.new_field(3), 8e5, spec)
