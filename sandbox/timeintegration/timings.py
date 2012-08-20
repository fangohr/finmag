import commands, time, sys

files = ['macrospin_odeint_nojac', 'macrospin_odeint_jac', 'macrospin_ode_nojac', 'macrospin_ode_jac']
names = ['odeint', 'odeint with jacobian', 'ode', 'ode with jacobian']

for nr, f in enumerate(files):
    cmd = 'python %s.py' % (f)

    # Run all scripts once before timing starts, to avoid compiler timing.
    status, output = commands.getstatusoutput(cmd)
    if status != 0:
        sys.stderr.write(output + '\n')
        sys.exit(status)
    t0 = time.time()
    for i in range(10):
        status, output = commands.getstatusoutput(cmd)
        if status != 0:
            sys.stderr.write(output + '\n')
            sys.exit(status)
    t1 = time.time()
    print 'Time using %s: %.2f sec.' % (names[nr], t1-t0)

