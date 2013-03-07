all:
	echo "Please call one of the makefile targets directly. For example: 'make averages_nmag5.txt'"

averages_nmag5.txt:
	nsim run_nmag5.py
	ncol run_nmag5_dat.ndt time m_Py_0 m_Py_1 m_Py_2 > $@

clean:
	rm -f run_nmag5_dat.ndt
