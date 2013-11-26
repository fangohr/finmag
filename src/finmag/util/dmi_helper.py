import numpy as np


def find_skyrmion_center_2d(fun, point_up=False):
    """
    Find the centre the skyrmion, suppose only one skyrmion
    and only works for 2d mesh.
    
     `fun` accept a dolfin function.
     `point_up` : the core of skyrmion,  points up or points down.
      
    """
    
    V = fun.function_space()

    mesh = V.mesh()
    coods = V.dofmap().tabulate_all_coordinates(mesh).reshape(3,-1)[0]
    coods.shape = (-1, mesh.topology().dim())

    xs = coods[:,0]
    ys = coods[:,1]
    
    mxys = fun.vector().array().reshape(3,-1)
    mzs = mxys[2]
    
    if point_up:
        mzs = - mxys[2]
    
    mins = [i for i,u in enumerate(mzs) if u<-0.9 ]
    
    xs_max = np.max(xs[mins])
    xs_min = np.min(xs[mins])
    
    ys_max = np.max(ys[mins])
    ys_min = np.min(ys[mins])
    
    xs_refine = np.linspace(xs_min, xs_max, 101)
    ys_refine = np.linspace(ys_min, ys_max, 101)
    
    coods_refine = np.array([(x, y) for x in xs_refine for y in ys_refine])
    mzs_refine = np.array([fun(xy)[2] for xy in coods_refine])
    
    min_id = np.argmin(mzs_refine)
    if point_up:
        min_id = np.argmax(mzs_refine)
    
    center = coods_refine[min_id]
    
    return center[0],center[1]
