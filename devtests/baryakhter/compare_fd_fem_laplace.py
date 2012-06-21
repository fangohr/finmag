from dolfin import *
import numpy as np


def Laplace_FD_2d(mesh,u_e,gridm,gridn):
    data=mesh.coordinates()
    m,n=data.shape
    assert((gridm+1)*(gridn+1)==m)
    x=data[:,0]
    y=data[:,1]
    x=x.reshape((gridm+1,gridn+1),order='F')
    y=y.reshape((gridm+1,gridn+1),order='F')
    u=u_e.vector().array().reshape((gridm+1,gridn+1,2),order='F')

    dx=(x[-1][0]-x[0][0])/gridm
    dy=(y[0][-1]-y[0][0])/gridn
    print 'dx*dx=',dx*dx,'dy*dy=',dy*dy
    diff=np.zeros((gridm+1,gridn+1,2))
 
    for i in range(gridm+1):
        for j in range(gridn+1):
            if i>0:
                diff[i,j,0]+=(u[i-1,j,0]-u[i,j,0])/(dx*dx)
                diff[i,j,1]+=(u[i-1,j,1]-u[i,j,1])/(dx*dx)
            if j>0:
                diff[i,j,0]+=(u[i,j-1,0]-u[i,j,0])/(dy*dy)
                diff[i,j,1]+=(u[i,j-1,1]-u[i,j,1])/(dy*dy)
            if i<gridm:
                diff[i,j,0]+=(u[i+1,j,0]-u[i,j,0])/(dx*dx)
                diff[i,j,1]+=(u[i+1,j,1]-u[i,j,1])/(dx*dx)
            if j<gridn:
                diff[i,j,0]+=(u[i,j+1,0]-u[i,j,0])/(dy*dy)
                diff[i,j,1]+=(u[i,j+1,1]-u[i,j,1])/(dy*dy)



    diff=diff.reshape(2*m,order='F')
 
 
    print diff
    Vv = VectorFunctionSpace(mesh, 'Lagrange', 1)
    diff_u=Function(Vv)
    diff_u.vector()[:]=diff[:]
    return diff_u
    
def Laplace_FEM(mesh,u_e):
    V = FunctionSpace(mesh, 'Lagrange', 1)
    V3 = VectorFunctionSpace(mesh,'CG',1)
    grad_u = project(grad(u_e))
    tmp=project(div(grad_u))
    return tmp



if __name__=='__main__':
    gridm,gridn=100,1
    mesh = Rectangle(0, 0, 10*np.pi, 10, gridm, gridn, 'left')
    V = FunctionSpace(mesh, 'Lagrange', 1)
    V3 = VectorFunctionSpace(mesh,'CG',1)


    v0 = Expression(('-cos(x[0])','sin(x[0])'))
    v0_e=Expression(('cos(x[0])','-sin(x[0])'))

   
   
 
    v = interpolate(v0, V3)
    v_e = interpolate(v0_e, V3)


    fd=Laplace_FD_2d(mesh,v,gridm,gridn)
   
    fem=Laplace_FEM(mesh,v)
    diff=Function(V3)
    diff.vector()[:]=fem.vector().array()-v_e.vector().array()
    print diff.vector().array().reshape((gridm+1,gridn+1,2),order='F')

    diff_fd=Function(V3)
    diff_fd.vector()[:]=fd.vector().array()-v_e.vector().array()    

    plot(fem)
    plot(diff)
    plot(v_e)
    plot(fd)
    plot(diff_fd)
    interactive()
