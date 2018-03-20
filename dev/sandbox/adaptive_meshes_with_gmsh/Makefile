MESHNAME ?= disk

all:

mesh:
	gmsh -3 $(MESHNAME).geo -3 -optimize -optimize_netgen -o $(MESHNAME).msh
	dolfin-convert $(MESHNAME).msh $(MESHNAME).xml
	python -c "import dolfin as df; mesh = df.Mesh('$(MESHNAME).xml'); df.plot(mesh); df.interactive()"
