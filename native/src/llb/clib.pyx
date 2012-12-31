import numpy as np
cimport numpy as np

cdef extern from "sllg.h":
	ctypedef struct ode_solver:
		pass
	
	ode_solver *create_ode_plan()
	void init_solver(ode_solver *s,double *alpha, double *T, double *V, double *Ms, int nxyz)
	void init_solver_parameters(ode_solver *s, double gamma, double dt, double c)
	void finalize_ode_plan(ode_solver *plan)
	void run_step1(ode_solver *s, double *m, double *h, double *m_pred)
	void run_step2(ode_solver *s, double *m_pred, double *h, double *m)
	
		
cdef class RK2S(object):
	cdef ode_solver * _c_plan
	cdef double dt
	cdef update_fun
	cdef np.ndarray spin
	cdef np.ndarray pred_m
	cdef np.ndarray field
		
	def __cinit__(self,nxyz,
				np.ndarray[double, ndim=1, mode="c"] spin,
				np.ndarray[double, ndim=1, mode="c"] alpha,
				np.ndarray[double, ndim=1, mode="c"] T,
				np.ndarray[double, ndim=1, mode="c"] V,
				np.ndarray[double, ndim=1, mode="c"] Ms,
				update_fun):
		self.spin=spin
		self.update_fun=update_fun
		self.pred_m=np.zeros(3*nxyz,dtype=np.float)
		
		self._c_plan = create_ode_plan()
		if self._c_plan is NULL:
			raise MemoryError()
		
		init_solver(self._c_plan,&alpha[0],&T[0],&V[0],&Ms[0],nxyz)
		
	def __dealloc__(self):
		if self._c_plan is not NULL:
			finalize_ode_plan(self._c_plan)
			self._c_plan = NULL
		
	def setup_parameters(self,gamma,dt,c):
		init_solver_parameters(self._c_plan, gamma,dt,c)
				
	def run_step(self, np.ndarray[double, ndim=1, mode="c"] field):
		cdef np.ndarray[double, ndim=1, mode="c"] spin=self.spin
		cdef np.ndarray[double, ndim=1, mode="c"] pred_m=self.pred_m
		
		self.update_fun(spin)
		run_step1(self._c_plan,&spin[0],&field[0],&pred_m[0])
		
		self.update_fun(self.pred_m)
		run_step2(self._c_plan,&pred_m[0],&field[0],&spin[0])
					

	
	
		
