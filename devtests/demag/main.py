"""Main Module to run the other modules from"""
from solver_nitsche import *
from solver_gcr import *
from prob_trunc_testcases import *
import prob_fembem_testcases as pft
from dolfin import *

fembemproblem = pft.MagUnitSphere(10)
gcrsolver = GCRFemBemDeMagSolver(fembemproblem)
gcrsol = gcrsolver.solve()
gcr = project(gcrsol,FunctionSpace(gcrsol.function_space().mesh(),"CG",1))
gcrsolver.save_function(gcr,"potential")
##print gcrsolver.Hdemag(0,0,0)
##print gcrsolver.Hdemag(0,0.5,0)
##print gcrsolver.Hdemag(0,0,0.5)
##print gcrsolver.Hdemag(0.5,0,0)

