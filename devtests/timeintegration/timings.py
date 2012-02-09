import commands, time

files = ['macrospin_odeint_nojac', 'macrospin_odeint_jac', 'macrospin_ode_nojac', 'macrospin_ode_jac']
names = ['odeint', 'odeint with jacobian', 'ode', 'ode with jacobian']
for nr, f in enumerate(files):
    cmd = 'python %s.py' % f
    t0 = time.time()
    for i in range(15):
        status, output = commands.getstatusoutput(cmd)
    t1 = time.time()
    print 'Time using %s: %.2f sec.' % (names[nr], t1-t0)

