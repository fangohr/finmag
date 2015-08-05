import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt

import pytest
import numpy as np
import dolfin as df
from exchange import ExchangeDG
from finmag.energies.exchange import Exchange
from finmag.util.consts import mu0
from finmag.util.meshes import box
from finmag.field import Field


@pytest.fixture(scope = "module")
def fixt():
    """
    Create an Exchange object that will be re-used during testing.

    """
    mesh = df.UnitCubeMesh(10, 10, 10)
    S3 = df.VectorFunctionSpace(mesh, "DG", 0)
    Ms = 1
    A = 1
    m = df.Function(S3)
    exch = ExchangeDG(A)
    exch.setup(S3, m, Ms)
    return {"exch": exch, "m": m, "A": A, "S3": S3, "Ms": Ms}


def test_interaction_accepts_name():
    """
    Check that the interaction accepts a 'name' argument and has a 'name' attribute.
    """
    exch = Exchange(13e-12, name='MyExchange')
    assert hasattr(exch, 'name')


def test_there_should_be_no_exchange_for_uniform_m(fixt):
    """
    Check that exchange field and energy are 0 for uniform magnetisation.

    """
    TOLERANCE = 1e-6
    fixt["m"].assign(df.Constant((1, 0, 0)))

    H = fixt["exch"].compute_field()
    print "Asserted zero exchange field for uniform m = (1, 0, 0), got H =\n{}.".format(H.reshape((3, -1)))
    assert np.max(np.abs(H)) < TOLERANCE


def plot_m(mesh,xs,m_an,f_dg,f_cg,name='compare.pdf'):
    
    fig=plt.figure()
    
    plt.plot(xs,m_an,'--',label='Analytical')
    
    dg=[]
    cg=[]
    for x in xs:
        dg.append(f_dg(x,1.0,1.0)[2])
        cg.append(f_cg(x,1.0,1.0)[2])
    np.savetxt('xs.txt',np.array(xs))
    np.savetxt('cg.txt', np.array(cg))
    np.savetxt('dg.txt', np.array(dg))
    np.savetxt('analytical.txt', np.array(m_an))

    plt.plot(xs,dg,'.-',label='dg')
    plt.plot(xs,cg,'^-',label='cg')
    plt.xlabel('x')
    plt.ylabel('Field')
    plt.legend(loc=8)
    fig.savefig(name)
 
 
def map_dg2cg(mesh, fun_dg):
    
    CG = df.VectorFunctionSpace(mesh, 'CG', 1, dim=3)
    DG = df.VectorFunctionSpace(mesh, 'DG', 0, dim=3)
    fun_cg = df.Function(CG)
    
    u = df.TrialFunction(DG)
    v = df.TestFunction(CG)
    L = df.assemble(df.dot(v, df.Constant([1,1,1]))*df.dx).array()
    a = df.dot(u, v) * df.dx
    A = df.assemble(a).array()

    m = fun_dg.vector().array()

    fun_cg.vector()[:]=np.dot(A,m)/L
    
    return fun_cg
    


if __name__ == "__main__":
    #mesh = df.IntervalMesh(10, 0, 2*np.pi)
    #
    #mesh = df.RectangleMesh(0,0,2*np.pi,1,10,1)
    
    mesh = box(0,0,0,100,5,5, maxh=2.0)
    #mesh = df.BoxMesh(0,0,0,100,5,5,50, 2, 2)
    #mesh = box(0,0,0,2*np.pi,0.5,0.5, maxh=0.3) 


    df.plot(mesh)
    df.interactive()
    DG = df.VectorFunctionSpace(mesh, "DG", 0, dim=3)
    C = 1.3
    expr = df.Expression(('0', 'sin(2*pi*x[0]/100)','cos(2*pi*x[0]/100)'))
    Ms = 8.6e5
    m = df.interpolate(expr, DG)
    
    exch = ExchangeDG(C)
    exch.setup(DG, m, Ms, unit_length=1)
    f = exch.compute_field()
    
    xs=np.linspace(1,99,31)
    m_an= -1.0*2*C/(mu0*Ms)*(2*np.pi/100)**2*np.cos(2*np.pi/100*xs)
    
    
    field=df.Function(DG)
    field.vector().set_local(f)
    field=map_dg2cg(mesh, field)
    
    
    S3 = df.VectorFunctionSpace(mesh, "CG", 1, dim=3)
    S1 = df.FunctionSpace(mesh, "CG", 1)
    m2 = Field(S3, expr)
    exch = Exchange(C)
    exch.setup(m2, Field(S1,Ms), unit_length=1)
    f = exch.compute_field()
    field2 = df.Function(S3)
    field2.vector().set_local(f)

    plot_m(mesh,xs,m_an,field,field2)

    df.plot(field)
    

    df.plot(field2)
    df.interactive()

    
    