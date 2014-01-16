import os
import textwrap
import dolfin as df
import numpy as np
from demag import Demag


class FixedEnergyDW(object):
    def __init__(self, left=(1, 0, 0), right=(-1, 0, 0), repeat_time=5):
        self.left = left
        self.right = right
        self.repeat_time = repeat_time
        self.in_jacobian = False

    def write_xml(self, filename, coordinates, cells):
        f = open(filename, 'w')
        f.write("""<?xml version="1.0"?>\n""")
        f.write("""<dolfin xmlns:dolfin="http://fenicsproject.org">\n""")
        f.write("""   <mesh celltype="tetrahedron" dim="3">\n""")
        f.write("""     <vertices size="%d">\n"""%len(coordinates))
        for i in range(len(coordinates)):
            f.write("""       <vertex index="%d" x="%0.12f" y="%0.12f" z="%0.12f"/>\n"""%(
                    i,
                    coordinates[i][0],
                    coordinates[i][1],
                    coordinates[i][2]))
        f.write("""     </vertices>\n""")
        f.write("""     <cells size="%d">\n"""%len(cells))
        for i in range(len(cells)):
            f.write("""       <tetrahedron index="%d" v0="%d" v1="%d" v2="%d" v3="%d"/>\n"""%(
                    i,
                    cells[i][0],
                    cells[i][1],
                    cells[i][2],
                    cells[i][3]))
        f.write("""     </cells>\n""")
        f.write("""  </mesh>\n</dolfin>""")

    def bias_mesh(self,step):
        cds=np.array(self.mesh.coordinates())
        cells=np.array(self.mesh.cells())

        cells+=len(cds)
        cells=np.concatenate((self.mesh.cells(),cells))

        cds[:,0]+=self.xlength*step
        cds=np.concatenate((self.mesh.coordinates(), cds))

        return cells,cds

    def setup(self, S3, m, Ms, unit_length=1):
        self.S3=S3
        self.mesh=S3.mesh()
        self.Ms=Ms
        n=self.mesh.num_vertices()
        self.tmp_field=np.zeros(6*n)
        self.field=np.zeros((n,3))
        self.init_m=np.zeros((2*n,3))

        c=self.mesh.coordinates()
        self.xlength=np.max(c[:,0])-np.min(c[:,0])

        self.__compute_field()
        tmp=self.tmp_field.reshape((3,-1),order='C')
        self.field=np.array(tmp[:,:n])
        self.field.shape=(1,-1)
        self.field=self.field[0]

    def __compute_field(self):
        n=self.mesh.num_vertices()
        self.init_m[:n,0]=1
        self.init_m[n:,:]=self.left

        for i in range(-self.repeat_time,0):
            cells,cds=self.bias_mesh(i-1e-10)
            filename="mesh_%d.xml"%i
            self.write_xml(filename,cds,cells)

            demag=Demag(solver='Treecode')
            mesh=df.Mesh(filename)
            Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)

            dg = df.FunctionSpace(mesh, "DG", 0)
            Ms_tmp=df.Function(dg)
            Ms_list=list(self.Ms.vector().array())
            Ms_tmp.vector().set_local(np.array(Ms_list+Ms_list))


            m=df.Function(Vv)
            tmp_init_m=self.init_m.reshape((1,-1),order='F')[0]
            m.vector().set_local(tmp_init_m)
            demag.setup(Vv,m,Ms_tmp)
            self.tmp_field+=demag.compute_field()

            os.remove(filename)

        self.init_m[:n,0]=-1
        self.init_m[n:,:]=self.right

        for i in range(1,self.repeat_time+1):
            cells,cds=self.bias_mesh(i+1e-10)
            filename="mesh_%d.xml"%i
            self.write_xml(filename,cds,cells)

            demag=Demag(solver='Treecode')
            mesh=df.Mesh(filename)
            Vv = df.VectorFunctionSpace(mesh, 'Lagrange', 1)

            dg = df.FunctionSpace(mesh, "DG", 0)
            Ms_tmp=df.Function(dg)
            Ms_list=list(self.Ms.vector().array())
            Ms_tmp.vector().set_local(np.array(Ms_list+Ms_list))

            m=df.Function(Vv)
            tmp_init_m=self.init_m.reshape((1,-1),order='F')[0]
            m.vector().set_local(tmp_init_m)
            demag.setup(Vv,m,Ms_tmp)
            self.tmp_field+=demag.compute_field()

            os.remove(filename)

    def compute_field(self):
        return self.field



if __name__=='__main__':
    mesh = df.BoxMesh(0, 0, 0, 500, 20, 5, 100, 4, 1)

    dw=FixedEnergyDW(repeat_time=5)
    S3 = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    m=df.Function(S3)
    dw.setup(S3, 1, 8.6e5, unit_length=1)
    m.vector().set_local(dw.compute_field())
    print dw.compute_field().reshape((3,-1))
    for x in range(100):
        print x*5+2.5,m(x*5+2.5,17.5,2.5)
