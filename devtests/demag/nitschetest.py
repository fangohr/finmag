"The demagnetisation field for a unit interval is computed given a Mag field"
"(1,0,0) and an approximation of the infinite domain"
"Nitsche's method is used to force the jump in the normal derivative accross"
"the boundry of the magnetic body" 

__author__ = "Gabriel Balaban"
__copyright__ = __author__
__license__  = "GNU GPL Version 3 or any later version"

from dolfin import *
from interiorboundary import *
import math
import numpy as np
import pylab as pl

#Parameter to tweak the method
gamma = 0.001

n = 50
mesh = UnitInterval(n)
h = 1.0/n

V = FunctionSpace(mesh,"CG",2)
L = FunctionSpace(mesh,"CG",2)
W = MixedFunctionSpace((V,V))
u0,u1 = TestFunctions(W)
v0,v1 = TrialFunctions(W)
sol = Function(W) 
phi0 = Function(V)
phi1 = Function(V)
phitot = Function(V)

#Define the magnetisation
M = interpolate(Expression("1"),V)

#Define the Magnetic domain
r = 0.1 #Radius of magnetic Core
class MagCore(SubDomain):
    def inside(self, x, on_boundary):
        return x[0]< 0.5 + r + DOLFIN_EPS and x[0] > 0.5 - r - DOLFIN_EPS
    
corefunc = MeshFunction("uint", mesh, 1)
corefunc.set_all(0)
coredom = MagCore()  
coredom.mark(corefunc,1)
#Measure for integration over the core

###################################
#Define the magnetic boundary layer
###################################
intbound = InteriorBoundary(mesh)
coremesh = SubMesh(mesh,corefunc,1)
intbound.create_boundary(coremesh)
intfacet = intbound.boundaries[0]
##coreboundmeshfacets = create_intbound(mesh,MagCore())
N = FacetNormal(coremesh)
dSC = dS(2)

coreboundmesh = BoundaryMesh(coremesh)
##print "Mesh coordinates", mesh.coordinates()
##print "Submesh coordinates", coremesh.coordinates()
##print "Number of facets in submesh", coreboundmesh.num_cells()

#Define jumps and averages accross the boundary
jumpu = u1('-') - u0('+')
avggradu = (grad(u1('-')) + grad(u0('+')))*0.5
jumpv = (v1('-') - v0('+'))
avgv = (v1('-') + v0('+'))*0.5
avggradv = (grad(v1('-')) + grad(v0('+')))*0.5

#Forms for Poisson with Nitsche Method
a0 = dot(grad(u0),grad(v0))*dx(0) #Vacuum 
a1 = dot(grad(u1),grad(v1))*dx(1) #Core

#right hand side
f = (div(M)*v1)*dx(1)   #Source term in core
f += (dot(M('-'),N('+'))*avgv )*dSC  #Presribed outer normal derivative
#Cross terms on the interior boundary
c = (-dot(avggradu,N('+'))*jumpv - dot(avggradv,N('+'))*jumpu + gamma*(1/h)*jumpu*jumpv)*dSC  

a = a0 + a1 + c

#Dirichlet BC at our approximate boundary
dbc = DirichletBC(W.sub(0),0.0,"on_boundary")

A = assemble(a,cell_domains = corefunc, interior_facet_domains = intfacet)
F = assemble(f,cell_domains = corefunc, interior_facet_domains = intfacet)

dbc.apply(A)
dbc.apply(F)
A.ident_zeros()
##print "Matrix"
##print A.array()
##print "Vector f"
##print F.array()
##md.diagnose(A.array(),"Jacobian")
solve(A, sol.vector(),F)
solphi0,solphi1 = sol.split()

##print "Lagrange mult DOFS", phi.vector().array()
#demag = project(grad(phi),VV)
##print "Solution DOFS", phi.vector().array()
phi0.assign(solphi0)
####plot(phi0, title = "phi0")
phi1.assign(solphi1)
####plot(phi1, title = "phi1")
#phitot = phi0 + phi1
phitot.vector()[:] = phi0.vector() + phi1.vector()
plot(phitot, title = "phi total")
interactive()

###############################
#Test the Solution
###############################
#1 Test dirichlet boundary condition on outside
one = interpolate(Constant(1),V)
a = abs(phitot)*ds
c = one*ds
L1error = assemble(a)/assemble(c)
print "Average Error in Outer Dirichlet BC", L1error

#2 Test Continuity accross the interior boundary
one = interpolate(Constant(1),V)
jumpphi = phi1('-') - phi0('+')
a1 = abs(jumpphi)*dSC
a2 = abs(jump(phitot))*dSC
c = one('-')*dSC
L1error1 = assemble(a1,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
L1error2 = assemble(a2,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
print "Average Error in continuity in inner boundary for phi1 and phi2", L1error1
print "Average Error in continuity in inner boundary for phi total", L1error2


#3 Test jump in normal derivative across the interior boundary
one = interpolate(Constant(1),V)
jumpphinor = dot(grad(phi1('-') - phi0('+')),N('+'))
a1 = abs(jumpphinor - dot(M,N)('-'))*dSC
a2 = abs(dot(jump(grad(phitot)),N('+')) - dot(M,N)('+'))*dSC
c = one('-')*dSC
L1error1 = assemble(a1,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
L1error2 = assemble(a2,interior_facet_domains = intfacet )/assemble(c,interior_facet_domains = intfacet)
print "Average Error in jump in normal derivative in inner boundary for phi1 and phi2", L1error1
print "Average Error in jump in normal derivative in inner boundary for phi total", L1error1



###############################
#Gives the Volume of the surface of the magnetic core
###############################
#one = interpolate(Constant(1),V)
#volform = one('-')*dSC
#vol = assemble(volform,interior_facet_domains = intfacet)
#print "Volume of magnetic core using generated boundary", vol

##W = FunctionSpace(coreboundmesh,"CG",1)
##one = interpolate(Constant(1),W)
##volform = one*dx
##vol = assemble(volform)
##print "Volume of magnetic core usig boundary of submesh", vol
#print "Exact surface area of a sphere with same radius as magnetic core", 2

###############################
#Post process
##############################
##demagfile = File("demag.pvd")
##demagfile << sol
##print "value of Demag field on the boundary point (1,0,0)"
##print "x", sol[0]((1,0,0))
##print "y", sol[1]((1,0,0))
##print "z", sol[2]((1,0,0))
##print "somewhere else", sol[0]((0.1,0,0))

