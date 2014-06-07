import dolfin as df
import numpy as np
import inspect
from aeon import default_timer
from finmag.native import sundials
import finmag.native.llb as native_llb
from finmag.energies import Zeeman
from finmag.energies import Demag
from finmag.physics.llb.exchange import Exchange
from finmag.physics.llb.material import Material
from finmag.util import helpers
from finmag.util.vtk_saver import VTKSaver
from finmag.util.fileio import Tablewriter
from finmag.scheduler import scheduler, derivedevents
from finmag.util.pbc2d import PeriodicBoundary2D

import logging
log = logging.getLogger(name="finmag")

ONE_DEGREE_PER_NS = 17453292.5  # in rad/s


class LLB(object):
    def __init__(self, mat, method='RK2b',name='unnamed',pbc2d=None):
        self.material = mat
        self._m = mat._m
        self.m = self._m.vector().array()
        self.S1 = mat.S1
        self.S3 = mat.S3
        self.mesh=self.S1.mesh()

        self.dm_dt = np.zeros(self.m.shape)
        self.H_eff = np.zeros(self.m.shape)

        self.time_scale=1e-9
        self.method=method
        self.pbc2d=pbc2d

        self.set_default_values()
        self.interactions.append(mat)


        if self.pbc2d:
            self.pbc2d=PeriodicBoundary2D(self.S3)

        self.name = name
        self.sanitized_name = helpers.clean_filename(name)

        self.logfilename = self.sanitized_name + '.log'
        self.ndtfilename = self.sanitized_name + '.ndt'

        helpers.start_logging_to_file(self.logfilename, mode='w', level=logging.DEBUG)
        self.scheduler = scheduler.Scheduler()

        self.domains =  df.CellFunction("uint", self.mesh)
        self.domains.set_all(0)
        self.region_id=0

        self.tablewriter = Tablewriter(self.ndtfilename, self, override=True)

        self.overwrite_pvd_files = False
        self.vtk_export_filename = self.sanitized_name + '.pvd'
        self.vtk_saver = VTKSaver(self.vtk_export_filename,overwrite=True)

        self.scheduler_shortcuts = {
            'save_ndt' : LLB.save_ndt,
            'save_vtk' : LLB.save_vtk,
            }



    def set_default_values(self):

        self.alpha = self.material.alpha
        self.gamma_G = 2.21e5 # m/(As)
        self.gamma_LL = self.gamma_G/(1. + self.alpha**2)

        self.t = 0.0 # s
        self.do_precession = True

        self.vol = df.assemble(df.dot(df.TestFunction(self.S3),
                                      df.Constant([1, 1, 1])) * df.dx).array()
        self.real_vol = self.vol*self.material.unit_length**3

        self.nxyz = self.mesh.num_vertices()
        self._alpha = np.zeros(self.nxyz)

        self.pins=[]
        self._pre_rhs_callables = []
        self._post_rhs_callables = []
        self.interactions = []

    def set_up_solver(self, reltol=1e-8, abstol=1e-8, nsteps=10000):
        integrator = sundials.cvode(sundials.CV_BDF, sundials.CV_NEWTON)
        integrator.init(self.sundials_rhs, 0, self.m)
        integrator.set_linear_solver_sp_gmr(sundials.PREC_NONE)
        integrator.set_scalar_tolerances(reltol, abstol)
        integrator.set_max_num_steps(nsteps)

        self.integrator = integrator
        self.method = 'cvode'

    def set_up_stochastic_solver(self, dt=1e-13,using_type_II=True):

        self.using_type_II = using_type_II

        M_pred=np.zeros(self.m.shape)

        integrator = native_llb.StochasticLLBIntegrator(
                                    self.m,
                                    M_pred,
                                    self.material.Ms,
                                    self.material.T,
                                    self.material.real_vol,
                                    self.pins,
                                    self.stochastic_rhs,
                                    self.method)

        self.integrator = integrator
        self._seed=np.random.random_integers(4294967295)
        self.dt=dt


    @property
    def t(self):
        return self._t*self.time_scale

    @t.setter
    def t(self,value):
        self._t=value/self.time_scale

    @property
    def dt(self):
        return self._dt*self.time_scale

    @dt.setter
    def dt(self, value):
        self._dt=value/self.time_scale
        log.info("dt=%g."%self.dt)
        self.setup_parameters()


    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed=value
        log.info("seed=%d."%self._seed)
        self.setup_parameters()

    def set_pins(self, nodes):
        pinlist=[]
        if hasattr(nodes, '__call__'):
            coords = self.mesh.coordinates()
            for i,c in enumerate(coords):
                if nodes(c):
                    pinlist.append(i)
        else:
            pinlist=nodes

        self._pins = np.array(pinlist, dtype="int")

        if self.pbc2d:
            self._pins=np.concatenate([self.pbc2d.ids_pbc,self._pins])

        if len(self._pins>0):
            self.nxyz=self.S1.mesh().num_vertices()
            assert(np.min(self._pins)>=0)
            assert(np.max(self._pins)<self.nxyz)


    def pins(self):
        return self._pins
    pins = property(pins, set_pins)


    def set_spatial_alpha(self, value):
        self._alpha[:]=helpers.scalar_valued_function(value,self.S1).vector().array()[:]


    def setup_parameters(self):
        self.integrator.set_parameters(self.dt,
                                       self.gamma_LL,
                                       self.alpha,
                                       self.material.Tc,
                                       self.seed,
                                       self.do_precession,
                                       self.using_type_II)

    def add(self,interaction):
        interaction.setup(self.material.S3,
                          self.material._m,
                          self.material.Ms0,
                          unit_length=self.material.unit_length)
        self.interactions.append(interaction)

        if interaction.__class__.__name__=='Zeeman':
            self.zeeman_interation=interaction
            self.tablewriter.entities['zeeman']={
                        'unit': '<A/m>',
                        'get': lambda sim: sim.zeeman_interation.average_field(),
                        'header': ('h_x', 'h_y', 'h_z')}

            self.tablewriter.update_entity_order()



    def compute_effective_field(self):
        self.H_eff[:]=0
        for interaction in self.interactions:
            self.H_eff += interaction.compute_field()

    def total_energy(self):
        #FIXME: change to the real total energy
        return 0


    def stochastic_rhs(self, y):

        self._m.vector().set_local(y)

        for func in self._pre_rhs_callables:
            func(self.t)

        self.compute_effective_field()

        for func in self._post_rhs_callables:
            func(self)


    def sundials_rhs(self, t, y, ydot):
        self.t = t
        self._m.vector().set_local(y)

        for func in self._pre_rhs_callables:
            func(self.t)

        self.compute_effective_field()

        default_timer.start("sundials_rhs", self.__class__.__name__)
        # Use the same characteristic time as defined by c

        native_llb.calc_llb_dmdt(self._m.vector().array(),
                                 self.H_eff,
                                 self.dm_dt,
                                 self.material.T,
                                 self.pins,
                                 self._alpha,
                                 self.gamma_LL,
                                 self.material.Tc,
                                 self.do_precession)


        default_timer.stop("sundials_rhs", self.__class__.__name__)

        for func in self._post_rhs_callables:
            func(self)

        ydot[:] = self.dm_dt[:]

        return 0

    def run_with_scheduler(self):
        if self.method=='cvode':
            run_fun=self.run_until_sundial
        else:
            run_fun=self.run_until_stochastic

        for t in self.scheduler:
            run_fun(t)
            self.scheduler.reached(t)
        self.scheduler.finalise(t)

    def run_until(self,time):

        # Define function that stops integration and add it to scheduler. The
        # at_end parameter is required because t can be zero, which is
        # considered as False for comparison purposes in scheduler.add.
        def StopIntegration():
            return False
        self.scheduler.add(StopIntegration, at=time, at_end=True)

        self.run_with_scheduler()

    def run_until_sundial(self, t):
        if t <= self.t:
            return

        self.integrator.advance_time(t,self.m)
        self._m.vector().set_local(self.m)
        self.t=t

    def run_until_stochastic(self, t):

        tp=t/self.time_scale

        if tp <= self._t:
            return
        try:
            while tp-self._t>1e-12:
                self.integrator.run_step(self.H_eff)
                self._m.vector().set_local(self.m)
                if self.pbc2d:
                    self.pbc2d.modify_m(self._m.vector())
                self._t+=self._dt
        except Exception,error:
            log.info(error)
            raise Exception(error)


        if abs(tp-self._t)<1e-12:
            self._t=tp
        log.debug("Integrating dynamics up to t = %g" % t)

    def m_average_fun(self,dx=df.dx):
        """
        Compute and return the average polarisation according to the formula
        :math:`\\langle m \\rangle = \\frac{1}{V} \int m \: \mathrm{d}V`

        """

        mx = df.assemble(self.material._Ms_dg*df.dot(self._m, df.Constant([1, 0, 0])) * dx)
        my = df.assemble(self.material._Ms_dg*df.dot(self._m, df.Constant([0, 1, 0])) * dx)
        mz = df.assemble(self.material._Ms_dg*df.dot(self._m, df.Constant([0, 0, 1])) * dx)
        volume = df.assemble(self.material._Ms_dg*dx)

        return np.array([mx, my, mz]) / volume
    m_average=property(m_average_fun)


    def save_m_in_region(self,region,name='unnamed'):

        self.region_id+=1
        helpers.mark_subdomain_by_function(region, self.mesh, self.region_id,self.domains)
        self.dx = df.Measure("dx")[self.domains]

        if name=='unnamed':
            name='region_'+str(self.region_id)

        region_id=self.region_id
        self.tablewriter.entities[name]={
                        'unit': '<>',
                        'get': lambda sim: sim.m_average_fun(dx=self.dx(region_id)),
                        'header': (name+'_m_x', name+'_m_y', name+'_m_z')}

        self.tablewriter.update_entity_order()

    def save_ndt(self):
        #log.debug("Saving average field values for simulation '{}'.".format(self.name))
        self.tablewriter.save()

    def schedule(self, func, *args, **kwargs):
        if isinstance(func, str):
            if func in self.scheduler_shortcuts:
                func = self.scheduler_shortcuts[func]
            else:
                msg = "Scheduling keyword '%s' unknown. Known values are %s" \
                    % (func, self.scheduler_shortcuts.keys())
                log.error(msg)
                raise KeyError(msg)

        func_args = inspect.getargspec(func).args
        illegal_argnames = ['at', 'after', 'every', 'at_end', 'realtime']
        for kw in illegal_argnames:
            if kw in func_args:
                raise ValueError(
                    "The scheduled function must not use any of the following "
                    "argument names: {}".format(illegal_argnames))

        at = kwargs.pop('at', None)
        after = kwargs.pop('after', None)
        every = kwargs.pop('every', None)
        at_end = kwargs.pop('at_end', False)
        realtime = kwargs.pop('realtime', False)

        self.scheduler.add(func, [self] + list(args), kwargs,
                at=at, at_end=at_end, every=every, after=after, realtime=realtime)

    def save_vtk(self, filename=None):
        """
        Save the magnetisation to a VTK file.
        """
        if filename != None:
            # Explicitly provided filename overwrites the previously used one.
            self.vtk_export_filename = filename

        # Check whether we're still writing to the same file.
        if self.vtk_saver.filename != self.vtk_export_filename:
            self.vtk_saver.open(self.vtk_export_filename, self.overwrite_pvd_files)

        self.vtk_saver.save_field(self._m, self.t)


    def relax(self, stopping_dmdt=ONE_DEGREE_PER_NS, dt_limit=1e-10,
              dmdt_increased_counter_limit=10000):
        """
        Run the simulation until the magnetisation has relaxed.

        This means the magnetisation reaches a state where its change over time
        at each node is smaller than the threshold `stopping_dm_dt` (which
        should be given in rad/s).

        """

        relax = derivedevents.RelaxationTimeEvent(self, stopping_dmdt, dmdt_increased_counter_limit, dt_limit)
        self.scheduler._add(relax)

        self.run_with_scheduler()
        self.integrator.reinit(self.t, self.m)

        self.scheduler._remove(relax)


if __name__ == '__main__':
    x0 = y0 = z0 = 0
    x1 = 500
    y1 = 10
    z1 = 100
    nx = 50
    ny = 1
    nz = 1
    mesh = df.BoxMesh(x0, y0, z0, x1, y1, z1, nx, ny, nz)

    mat = Material(mesh, name='FePt')
    mat.set_m((1, 0.2, 0))
    mat.T = 100
    mat.alpha=0.01

    sim = LLB(mat)
    #sim.set_up_solver()
    sim.set_up_stochastic_solver()

    sim.add(Zeeman((0, 0, 5e5)))
    sim.add(Exchange(mat))

    sim.add(Demag())

    #demag.demag.poisson_solver.parameters["relative_tolerance"] = 1e-8
    #demag.demag.laplace_solver.parameters["relative_tolerance"] = 1e-8


    max_time = 1 * np.pi / (sim.gamma_LL * 1e5)
    ts = np.linspace(0, max_time, num=100)

    mlist = []
    Ms_average = []
    for t in ts:
        print t
        sim.run_until(t)
        mlist.append(sim.m)
        df.plot(sim._m)



    df.interactive()
