import numpy as np

def v2d_xxx_from_xyz(v2d):
    """
    Return the vertex to dof map in the form

        v2d_xxx = [x0, x1, x2 , ... y0, y1, y2, ..., z0, z1, z2, ... ]
    
    using the vertex to dof map in the form (we are assuming this)

        v2d = [x0, y0, z0, x1 , y1, z1,  x2, y2, z2, ... ]

    """
    # Copy the original map (to not reference it)
    v2d_xxx = v2d.copy()
    
    # Reshape to have 3 rows with n numbers of columns
    v2d_xxx.shape=(3, -1)
    # Get rows,columns values
    (m, n) = v2d_xxx.shape
    
    # For every column of v2d_xxx , copy the corresponding values from
    # the xyz arrangement, for the 3 rows. For example:
    #
    # Row 0 (x values):
    #   [x0,   x1,    x2,      , ... , xi, ... ]  
    #           |      |               |
    #        v2d[3]   v2d[6]         v2d[3 * i]
    #
    # Row 1:
    #   [y0,   y1,    y2,      , ... , yi, ... ]  
    #           |      |               |
    #        v2d[4]   v2d[7]         v2d[3 * i + 1]
    #
    # Row 2: ...
    for i in range(n):
        v2d_xxx[0, i]=v2d[3 * i]
        v2d_xxx[1, i]=v2d[3 * i + 1]
        v2d_xxx[2, i]=v2d[3 * i + 2]
    # Return to the usual shape (see top) 
    v2d_xxx.shape=(-1, )
    return v2d_xxx

def d2v_fun(d2v, v2d):
    """

    Return an ordered full length dof to vertex map, in case of
    using Periodic Boundary Conditions

    The result is similar than the v2d_xxx function

    """
    # Copy the dof_to_vertex map
    a = d2v.copy()
    # Define one third of the length of the vortex to dof map
    # (with PBC, v2d is larger than the d2v map, since the
    # boundary vertexes are considered) 
    n = len(v2d) / 3
    # Now redefine every value of 'a' (magic? check this in the future)
    for i in range(len(a)):
        j = d2v[i]
        a[i]= (j % 3) * n + (j / 3)
    a.shape=(-1, )
    return a


import dolfin as df

df.parameters.reorder_dofs_serial = True

# Create a 1D mesh (interval) with 4 cells
# from 1 to 5
#
#   |------|------|------|------|
#   1                           5
# 
mesh = df.IntervalMesh(4, 1, 5)

# Define PBCs at the extremes
class PeriodicBoundary(df.SubDomain):
    # Define boundary at x = 1
    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return bool(x[0] < 1 + df.DOLFIN_EPS and x[0] > 1 - df.DOLFIN_EPS and on_boundary)

    # Map right boundary (H or x=5) to left boundary (G)
    # The function -map- maps a coordinate x in domain H to a coordinate y in the domain G
    def map(self, x, y):
        y[0] = x[0] - 4

# Create periodic boundary condition                                            
pbc = PeriodicBoundary()
# 3 dimensional vector space
V = df.VectorFunctionSpace(mesh, 'CG', 1, 3, constrained_domain=pbc)
# Set a vector field with values (x+0.1, x+0.2, x+0.3), defined on the interval
expression = df.Expression(['x[0]+0.1', 'x[0]+0.2', 'x[0]+0.3'])
f = df.interpolate(expression, V)

# Print the maps

# d2v should be smaller than v2d with PBCs
# For this case, the values from one boundary are not shown in
# d2v, since they are the same than the opposite boundary. Thus, for
# a 3d vector field, 3 values are omitted
d2v = df.dof_to_vertex_map(V)
# The full system map. Notice that boundary values are repeated
v2d = df.vertex_to_dof_map(V)

print 'd2v ', 'length =', len(d2v), d2v
print 'v2d ', 'length =', len(v2d), v2d
print 'v2d_old (xxx) ', v2d_xxx_from_xyz(v2d)

a = []
# Unordered collection:
b = set()

# Add all different index values from v2d to 'b' (no repetition) and 'a'
for x in v2d:
    if x not in b:
        b.add(x)
        a.append(x)
# Check if 'a' has the same length than the d2v (reduced) map 
assert(len(a)==len(d2v))
v2d_xyz = np.array(a)
print 'v2d_xyz (reduced)', v2d_xyz

print '\n'

# Get the values from the vector field (unordered)
# It does not show one boundary
a = f.vector().get_local()

# Map the values from vertex to dof. Since v2d is larger
# than the reduced vector, we can see that the extra values
# are the repeated boundaries
b = a[v2d]

# Get the ordered form of the v2d map
c = a[v2d_xxx_from_xyz(v2d)]

print '(Vector function values)'
print 'a=', a
print '(Mapped values with boundaries -- > v2d)'
print 'b=', b
print '(Ordered mapped values from the v2d --> v2d_xxx)'
print 'c=', c

print '\n'

print '(Mapped values , no boundaries)'
print 'b2=', a[v2d_xyz]

print '(Mapped values from the reduced v2d_xyz)'
print 'c2=', a[v2d_xxx_from_xyz(v2d_xyz)]

print 'Mapped values --> d2v'
print 'b[d2v]=', b[d2v]

print 'c[...]=', c[d2v_fun(d2v, v2d)]
