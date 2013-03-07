# Example motivated by dolfin mailing list (questions 221862): 
# http://answers.launchpad.net/dolfin/+question/221862 -- see last reply from Hake

# The entry on the list is not actually very
# informative, but the snippet below seems to show the key idea.
#
# I am gathering this here, to push forward the whole dofmap question
# that was introduced with dolfin 1.1 (HF, Feb 2013)


import dolfin as df

mesh = df.UnitSquareMesh(10,10)

V = df.FunctionSpace(mesh, 'CG', 1)

x = V.dofmap()

for cell in df.cells(mesh):
    print cell, V.dofmap().cell_dofs(cell.index())
	
