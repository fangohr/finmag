import numpy as np
import dolfin as df
import pylab

class Potential(object):
    """
    Idea: Have a potential, say 

    .. math::

       V(x) = 0.5*k*(f(x)-m(x))^2

    Have degrees of freedom m(x) that can adjust to this
    potential. The force field H is

    .. math::
    
      H = -dV/dm = k*(f(x)-m(x))


    For k=1 and m(x) =0, this simplifies to the force field being
    H(x)==f(x).

    Do some tests in this file, to demonstrate what 
    - the shape functions look like (vary order of basis functions)
    - the volume of the shape function means (and why we need to divide
      by it)
    
    Observations: the potential above is quadratic in m. if we use CG order=2 as the basis functiosn, then the resulting field H is the same as f.

    For CG order 1, a slight error occurs at the left - and right most 
    intervals.

    The project method seems to work fine (even the quadratic case in
    m seems t owork numerically exactly).

    This calculation is very similar to how the exchange, anisotropy
    and DMI field are calculated.

    """   
    
    def __init__(self, f, m, V):

        # Testfunction
        self.v = df.TestFunction(V)
        
        k=1
        # V
        self.E = 0.5*(k*(f-m)**2)*df.dx

        # Gradient
        self.dE_dM = df.Constant(-1.0)*df.derivative(self.E, m)

        # Volume
        self.vol = df.assemble(df.dot(self.v, 
            df.Constant(1)) * df.dx).array()

        # Store for later
        self.V = V
        self.m = m

        #method = 'project'
        method = 'box-assemble'
        if method=='box-assemble':
            self.__compute_field = self.__compute_field_assemble
        elif method == 'box-matrix-numpy':
            self.__setup_field_numpy()
            self.__compute_field = self.__compute_field_numpy
            raise NotImplementedError("Never tested for this example")
        elif method == 'box-matrix-petsc':
            self.__setup_field_petsc()
            self.__compute_field = self.__compute_field_petsc
            raise NotImplementedError("Never tested for this example")
        elif method=='project':
            self.__setup_field_project()
            self.__compute_field = self.__compute_field_project
        else:
            raise NotImplementedError("""Only methods currently implemented are
                                    * 'box-assemble', 
                                    * 'box-matrix-numpy',
                                    * 'box-matrix-petsc'  
                                    * 'project'""")

    def compute_field(self):
        """
        Compute the field.
        
         *Returns*
            numpy.ndarray
                The anisotropy field.       
        
        """
        H = self.__compute_field()
        return H
    
    def compute_energy(self):
        """
        Compute the anisotropy energy.

        *Returns*
            Float
                The anisotropy energy.

        """
        E=df.assemble(self.E)
        return E

    def __setup_field_numpy(self):
        """
        Linearise dE_dM with respect to M. As we know this is
        linear ( should add reference to Werner Scholz paper and
        relevant equation for g), this creates the right matrix
        to compute dE_dM later as dE_dM=g*M.  We essentially
        compute a Taylor series of the energy in M, and know that
        the first two terms (dE_dM=Hex, and ddE_dMdM=g) are the
        only finite ones as we know the expression for the
        energy.

        """
        g_form = df.derivative(self.dE_dM, self.m)
        self.g = df.assemble(g_form).array() #store matrix as numpy array  

    def __setup_field_petsc(self):
        """
        Same as __setup_field_numpy but with a petsc backend.

        """
        g_form = df.derivative(self.dE_dM, self.m)
        self.g_petsc = df.PETScMatrix()
        
        df.assemble(g_form,tensor=self.g_petsc)
        self.H_ani_petsc = df.PETScVector()

    def __setup_field_project(self):
        #Note that we could make this 'project' method faster by computing 
        #the matrices that represent a and L, and only to solve the matrix 
        #system in 'compute_field'(). IF this method is actually useful, 
        #we can do that. HF 16 Feb 2012

        #Addition: 9 April 2012: for this example, this seems to work.
        H_ani_trial = df.TrialFunction(self.V)
        self.a = df.dot(H_ani_trial, self.v) * df.dx
        self.L = self.dE_dM
        self.H_ani_project = df.Function(self.V)        

    def __compute_field_assemble(self):
        H_ani=df.assemble(self.dE_dM).array() / self.vol
        #H_ani=df.assemble(self.dE_dM).array() #/ self.vol
        return H_ani

    def __compute_field_numpy(self):
        Mvec = self.m.vector().array()
        H_ani = np.dot(self.g,Mvec)/self.vol
        return H_ani

    def __compute_field_petsc(self):
        self.g_petsc.mult(self.m.vector(), self.H_ani_petsc)
        H_ani = self.H_ani_petsc.array()/self.vol
        return H_ani

    def __compute_field_project(self):
        df.solve(self.a == self.L, self.H_ani_project)
        H_ani = self.H_ani_project.vector().array()
        return H_ani


def make_shapefunction(V,i,coeff=1):
    """
    V - function space

    i - node

    Return a (dolfin) function that, when plotted, shows the shape
    function associated with node i.

    """
    f = df.Function(V)
    f.vector()[:] *= 0
    f.vector()[i] = coeff
    print "i=%d, coeef=%f" % (i,coeff)

    return f


def plot_shapes_of_function(f):
    """For a given function f, create a plot that shows how the function
    is composed out of basis functions. 
    """
    import pylab
    V=f.function_space()
    mesh = V.mesh()
    coordinates = mesh.coordinates()
    xrange = max(coordinates),min(coordinates)
    n = len(coordinates)
    small_x = np.linspace(xrange[0],xrange[1],n*10)
    fcoeffs = f.vector().array()
    n = len(f.vector().array())
    for i in range(n):
        f_i = make_shapefunction(V,i,coeff=fcoeffs[i])
        f_i_values = [ f_i(x) for x in small_x ]
        pylab.plot(small_x,f_i_values,'--',label='i=%2d' % i)
    f_total_values = [ f(x) for x in small_x ]
    pylab.plot(small_x,f_total_values,label='superposition')
    f_total_mesh_values = [ f(x) for x in coordinates ]
    pylab.plot(coordinates,f_total_mesh_values,'o')
    pylab.legend()
    pylab.grid()
    pylab.show()

    return f


def plot_shapes_of_functionspace(V):
    """Create an overview of basis functions by plotting them.
    Expects (scalar) dolfin FunctionSpace object."""
    import pylab
    mesh = V.mesh()
    coordinates = mesh.coordinates()
    xrange = max(coordinates),min(coordinates)
    n = len(coordinates)
    small_x = np.linspace(xrange[0],xrange[1],n*100)
    n = len(f.vector().array())
    for i in range(n):
        f_i = make_shapefunction(V,i)
        f_i_values = [ f_i(x) for x in small_x ]

        pylab.plot(small_x,f_i_values,label='i=%2d' % i)
    pylab.legend()
    pylab.grid()
    pylab.show()

def compute_shape_function_volumes(V):
    v = df.TestFunction(V)
    Vs = df.assemble(1*v*df.dx)
    return Vs.array()

def test_shape_function_volumes(V):
    """Compace shape function volumes as computed in function 
    above, and by integrating the functions numerically with
    scipy.integrate.quad."""
    import scipy.integrate
    mesh = V.mesh()
    coordinates = mesh.coordinates()
    xrange = min(coordinates),max(coordinates)
    n = len(coordinates)
    shape_function_volumes = compute_shape_function_volumes(V)
    for i in range(n):
        f_i = make_shapefunction(V,i)
        shape_func_vol_quad = scipy.integrate.quad(f_i,xrange[0],xrange[1])[0]
        rel_error = (shape_function_volumes[i]-shape_func_vol_quad)/shape_function_volumes[i]
        print "i=%g, rel_error = %g" % (i,rel_error)
        assert rel_error <1e-15


if __name__=="__main__":
    minx = -3
    maxx = 2
    n = 5
    mesh = df.IntervalMesh(n,minx,maxx)
    #V = df.FunctionSpace(mesh,"CG",1)
    V = df.FunctionSpace(mesh,"CG",2)
    test_shape_function_volumes(V)
    m = df.Function(V)
    f = df.interpolate(df.Expression("-1*x[0]"),V)
    fproj = df.project(df.Expression("-1*x[0]"),V)
    plot_shapes_of_functionspace(V)
    print compute_shape_function_volumes(V)
    plot_shapes_of_function(f)

    p=Potential(f,m,V)
    Hcoeff=p.compute_field()
    H = df.Function(V)
    H.vector()[:]=Hcoeff[:]
    print mesh.coordinates()
    
    #compare f with force
    xs = np.linspace(minx,maxx,n*10)
    ftmp = [ f(x) for x in xs ]
    fprojtmp = [ fproj(x) for x in xs ]
    Htmp = [ H(x) for x in xs ]
    
    pylab.plot(xs,ftmp,label='f(x)')
    pylab.plot(xs,fprojtmp,'o',label='fproj(x)')
    pylab.plot(xs,Htmp,label='H(x)')
    pylab.legend()
    pylab.show()

    
