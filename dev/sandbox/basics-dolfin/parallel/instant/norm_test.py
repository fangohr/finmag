from dolfin import *

import instant

cpp_code = """
void dabla(dolfin::Vector& a, dolfin::Vector& b, double c, double d) {
    for (unsigned int i=0; i < a.size(); i++) {
        b.setitem(i, d*a[i] + c); 
    }
}
"""

include_dirs, flags, libs, libdirs = instant.header_and_libs_from_pkgconfig("dolfin")

headers= ["dolfin.h"]

func = instant.inline(cpp_code, system_headers=headers, include_dirs=include_dirs, libraries = libs, library_dirs = libdirs) 
#func = instant.inline(cpp_code, system_headers=headers)



if __name__ == '__main__':
    nx = ny = 1
    mesh = df.UnitSquareMesh(nx, ny)

    V = df.FunctionSpace(mesh, 'CG', 1)
    Vv = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    f = df.interpolate(df.Expression("0"),V)
    f1 = df.interpolate(df.Expression(("1","0","0")),Vv)
    f2 = df.interpolate(df.Expression(("0","1","0")),Vv)
    print 'a=',f1.vector().array()
    print 'b=',f2.vector().array()
    
    func(f1.vector(),f2,vector(),1.2,4.5)
    
    print 'a=',f1.vector().array()
    print 'b=',f2.vector().array()
    
