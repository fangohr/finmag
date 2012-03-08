
from dolfin import *
mesh=UnitCube(1,1,1)
V=VectorFunctionSpace(mesh,"CG",1)
a=Constant((0,0,1))
M=interpolate(Constant((0,0,8.6e5)),V)
Eform=2.3*dot(a,M)*dot(a,M)*dx #no warning
#Eform=2.2*dot(a,M)**2*dx        #warning

E=assemble(Eform)

"""When run the first time, the output is

Calling FFC just-in-time (JIT) compiler, this may take some time.
Summation index does not appear exactly twice: ?

"""
