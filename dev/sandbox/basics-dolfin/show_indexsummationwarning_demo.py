"""
Anisotropy Energy

Energy computed with two different, mathematically equivalent forms. The second
one produces the warning

    Calling FFC just-in-time (JIT) compiler, this may take some time.
    Summation index does not appear exactly twice: ?

when it is compiled.

"""
import dolfin as df

mesh = df.UnitCubeMesh(1, 1, 1)
V = df.VectorFunctionSpace(mesh,"CG",1)
a = df.Constant((0, 0, 1))
M = df.interpolate(df.Constant((0, 0, 8.6e5)), V)

print "Assembling first form."
Eform = 2.3 * df.dot(a, M) * df.dot(a, M) * df.dx
E = df.assemble(Eform)
print "E = ", E

print "Assembling second form."
Eform_alt = 2.3 * df.dot(a, M)**2 * df.dx
E_alt = df.assemble(Eform_alt)
print "E = ", E_alt

rel_diff = 100 * abs(E-E_alt)/E # in %
print "Difference is {} %.".format(rel_diff)
assert rel_diff < 0.1
