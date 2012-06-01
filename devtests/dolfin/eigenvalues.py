from dolfin import *

#mesh = UnitSphere(30)
#mesh = UnitCube(30,30,30)
#mesh = UnitCube(20,20,20)
mesh = UnitSquare(20,20)

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
eigensolver.solve(20)

r=[]
l=[]
for i in range(8):
	rr, c, rx, cx = eigensolver.get_eigenpair(i)
	r.append (rr)
	u = Function(V)
	u.vector()[:] = rx
	plot(u,title='mode %d' % i)

	#HF: not sure what the next lines are meant to do?
	e=project(grad(u),VectorFunctionSpace(mesh,"CG",1)) 
	l.append(assemble(dot(e,e)*dx)/assemble(u*u*dx)) 
	#from IPython import embed
	#embed()

print "Eigenvalues from solver:\n"
for i in r: print i

print "Eigenvalues from eigenfunctions:\n"
for i in l: print i

interactive() 
