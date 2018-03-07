#based on https://answers.launchpad.net/dolfin/+question/199058

from dolfin import *
import math

mesh = UnitSquareMesh(30,30)

lv = [c.volume() for c in cells(mesh)]
print "ratio of max and min volume: ", max(lv)/min(lv)

V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx
A = PETScMatrix()
assemble(a, tensor=A)

def u0(x,on_boundary): return on_boundary
bc = DirichletBC(V,Constant(0.0),u0)
bc.apply(A)
eigensolver = SLEPcEigenSolver(A)
eigensolver.parameters["spectrum"] = "smallest real" 
N=8

eigensolver.solve(N)

r=[]
l=[]
for i in range(N):
	rr, c, rx, cx = eigensolver.get_eigenpair(i)
	r.append (rr)
	u = Function(V)
	u.vector()[:] = rx

	#HF: not sure what the next two lines are meant to do?
	e=project(grad(u),VectorFunctionSpace(mesh,"CG",1)) 
	l.append(assemble(dot(e,e)*dx)/assemble(u*u*dx)) 

	plot(u,title='mode %d, EValue=%f EValue=%f' % (i,l[-1]/math.pi**2,rr/math.pi**2))

	#from IPython import embed
	#embed()

print "Eigenvalues from solver:\n"
for i in r: print i

print "Eigenvalues from eigenfunctions:\n"
for i in l: print i

interactive() 
