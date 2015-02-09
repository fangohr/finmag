def v2d_fun(v2d):
    v2d_old = v2d.copy()
    v2d_old.shape=(3,-1)
    (m,n) = v2d_old.shape
    for i in range(n):
        v2d_old[0,i]=v2d[3*i]
        v2d_old[1,i]=v2d[3*i+1]
        v2d_old[2,i]=v2d[3*i+2]
    v2d_old.shape=(-1,)
    return v2d_old

def d2v_fun(d2v, v2d):
    a = d2v.copy()
    n = len(v2d)/3
    for i in range(len(a)):
        j = d2v[i]
        a[i]= (j%3)*n+(j/3)
    a.shape=(-1,)
    return a


import dolfin as df
df.parameters.reorder_dofs_serial = False
mesh = df.IntervalMesh(4, 1, 5)
class PeriodicBoundary(df.SubDomain):
    def inside(self, x, on_boundary):
        return bool(x[0] < 1 + df.DOLFIN_EPS and x[0] >1-df.DOLFIN_EPS and on_boundary)
    def map(self, x, y):
        y[0] = x[0] - 4
# Create periodic boundary condition                                            
pbc = PeriodicBoundary()
V = df.VectorFunctionSpace(mesh, 'CG', 1, 3, constrained_domain=pbc)
expression = df.Expression(['x[0]+0.1', 'x[0]+0.2', 'x[0]+0.3'])
f = df.interpolate(expression, V)


d2v = df.dof_to_vertex_map(V)
v2d = df.vertex_to_dof_map(V)
print 'd2v',len(d2v),d2v
print 'v2d',len(v2d),v2d
print 'v2d_old',v2d_fun(v2d)

a = f.vector().get_local()
b = a[v2d]
c = a[v2d_fun(v2d)]

print 'a=',a
print 'b=',b
print 'c=',c
print 'b[d2v]=',b[d2v]
print 'c[...]=',c[d2v_fun(d2v,v2d)]
