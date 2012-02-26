all: run show

run: nmag dolfin

nmag:
	nsim run_nmag.py --clean
	ncol 1d time m_Py_0 m_Py_1 m_Py_2 > averages_ref.txt

dolfin:
	python run_dolfin.py

show:
	gnuplot "gnu.plot"
