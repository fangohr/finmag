
from dolfin import *
mesh=UnitCubeMesh(1,1,1)
V=VectorFunctionSpace(mesh,"CG",1)

M=interpolate(Constant((1,1,1)),V)

#E=dot(M,M)*dx
E=inner(M,curl(M))*dx
print "energy=",assemble(E)

dE_dM=derivative(E,M)
dE_dMvec=assemble(dE_dM)
print "vector=",dE_dMvec.array()

#curlM=project(somecurlform,V)

#print curlM.vector().array()


