import dolfin as df

#Test wether we can combine DG0 and CG1 expressions. Background:
#
#To allow M to vary as a function of space, we would like to write
#M = Ms*m where
#
# Ms is a DG0 space (with values on the centre of the cells) and
# m the normalised magnetisation vector defined on the nodes of the mesh.
#
# Seems to work. Hans & Weiwei, 19 April 2012.

mesh = df.RectangleMesh(0,0,1,1,1,1)

VV = df.VectorFunctionSpace(mesh,"CG",1,dim=3)
V  = df.FunctionSpace(mesh,"DG",0)

Ms = df.interpolate(df.Expression("1.0"),V)
m = df.interpolate(df.Expression(("2.0","0.0","0.0")),VV)
#m = df.interpolate(df.Expression("2.0"),VV)

L = df.dot(m,m)*Ms*df.dx
u = df.assemble(L)

print "expect u to be 4 (2*2*1):"
print "u=",u

