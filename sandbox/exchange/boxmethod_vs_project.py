import dolfin

from finmag.sim.exchange import Exchange

n=6
L=0.01
mesh = dolfin.Interval(n-1, 0, L)

V=dolfin.VectorFunctionSpace(mesh,"CG",1,dim=3)
Ms = 0.8e6 # A/m


left_right = 'Ms * (2*x[0]/L - 1)'
up_down = 'sqrt(Ms*Ms - Ms*Ms*(2*x[0]/L - 1)*(2*x[0]/L - 1))'
M = dolfin.interpolate(dolfin.Expression(('0','0','Ms'),L=L,Ms=Ms),V)

#playing with a uniform field
#M=dolfin.interpolate(dolfin.Constant([0,0,Ms]),V)
#M.vector()[1]=Ms
myM = M.vector().array()[:]


#print numpyM
C = 1.3e-11# J/m

exchange_box = Exchange(V,M,C,Ms,method='box')
exchange_project = Exchange(V,M,C,Ms,method='project')
box = exchange_box.compute_field()
project = exchange_project.compute_field()

myM.shape=(3,n)
print "M:\n",myM

box.shape=(3,n)
project.shape=(3,n)
import numpy
print "energy:\n",exchange_box.compute_energy()
print "Box:\n",numpy.round(box,decimals=3)
print "project:\n",numpy.round(project,decimals=3)
print "project-box:\n",numpy.round(project-box,decimals=3)


#oldMarray = M.vector().array()
#newM = dolfin.interpolate(
M.assign(dolfin.Expression((left_right,up_down,'0'),L=L,Ms=Ms))

#playing with a uniform field
print "=== Using uniform magnetisation ===="
#M.vector()[:]=dolfin.interpolate(dolfin.Constant([0,0,Ms]),V).vector().array()

box = exchange_box.compute_field()
project = exchange_project.compute_field()
print "energy:\n",exchange_box.compute_energy()
print "Box:\n",numpy.round(box,decimals=3)
print "project:\n",numpy.round(project,decimals=3)
print "project-box:\n",numpy.round(project-box,decimals=3)
print "box/project:\n",numpy.round(box/project,decimals=3)



#M.vector()[1]=Ms



