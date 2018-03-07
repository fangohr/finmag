import dolfin as df
import numpy as np
import time

from fastsum_lib import FastSum 


def exact(xs,xt,charge):
    res=np.zeros(len(xt))
    for j in range(len(xt)):
        for k in range(len(xs)):
            r=np.sqrt((xt[j,0]-xs[k,0])**2+(xt[j,1]-xs[k,1])**2+(xt[j,2]-xs[k,2])**2)
            res[j]+=1.0*charge[k]/r
    return res


if __name__ == '__main__':
	x0 = y0 = z0 = 0
	x1 = 1
	y1 = 1
	z1 = 1
	nx = 25
	ny = 25
	nz = 25

        
	mesh = df.Box(x0, y0, z0, x1, y1, z1, nx, ny, nz)
        n = 20
        mesh = df.UnitCubeMesh(n, n, n)
	mesh.coordinates()[:]*=1e-1
	number=mesh.num_vertices()
	print 'vertices number:',number
	
	xs=mesh.coordinates()
        
        n=10
        mesh = df.UnitCubeMesh(n, n, n)
	mesh.coordinates()[:]*=1e-1
        xt=mesh.coordinates()
        number=mesh.num_vertices()
	print 'target number:',number
        
        #density=np.array([1,2,3,4,5,6,7,8])*1.0
        fast_sum=FastSum(p=6,mac=0.5,num_limit=500)
        #xs=np.array([(2,7,0),(6.5,9,0),(6,8,0),(8,7,0),(4.9,4.9,0),(2,1,0),(4,2,0),(7,2,0)])*0.1
        
        #xt=np.array([(0,0,0)])*0.1
        density=np.random.random(len(xs))+1
    
        print xs,'\n',xt,'\n',density
        
    
	fast_sum.init_mesh(xs,xt)
        
        print xt
	
	fast_sum.update_charge(density)
        print density
	
	exact=np.zeros(len(xt))
	fast=np.zeros(len(xt))
        
        
        
        start=time.time()
	fast_sum.fastsum(fast)
        stop=time.time()
        print 'fast time:',stop-start
	print 'fast\n',fast
        
        start=time.time()
	fast_sum.exactsum(exact)
        stop=time.time()
        print 'exact time:',stop-start
	print 'exact\n',exact
        
	
	diff=np.abs(fast-exact)/exact
        print 'max:',np.max(diff)
        print diff
        print 'direct diff:\n',np.abs(fast-exact)
        
	
