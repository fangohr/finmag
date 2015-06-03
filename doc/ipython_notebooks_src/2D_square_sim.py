"""
Script to relax a skyrmion in the middle
of a 2D square system (trying to replicate a
Heisenberg model system)

IMPORTANT:
We are using the DMI expression from [1]

    D(m_x * dm_z/dx - m_z * dm_x/dx) +
    D(m_y * dm_z/dy - m_z * dm_y/dy) * df.dx

References:
[1] Rohart, S. and Thiaville A., Phys. Rev. B 88, 184422 (2013)

"""

# Material parameters from Rohart et al publication
# Ms = 1.1e6
# A = 16e-12


import os
import shutil
import argparse

# ARGUMENTS
parser = argparse.ArgumentParser(description='Relaxation of skyrmionic '
                                 'textures for 2D square films '
                                 'with interfacial DMI')
initial_state = parser.add_mutually_exclusive_group(required=True)

parser.add_argument('box_length', help='Length in nm',
                    type=float)

parser.add_argument('box_width', help='Width in nm',
                    type=float)

parser.add_argument('fd_max', help='Maximum edge length for the finite'
                    ' elements', type=float)

parser.add_argument('--D', help='DMI constant in units of 1e-3 * J m^{-2}',
                    type=float)

parser.add_argument('--A', help='Exchange constant in units of J m^{-1}',
                    type=float, default=1e-12)

parser.add_argument('--Ms', help='Saturation magnetisationin units of A / m',
                    type=float, default=1.1e6)

parser.add_argument('--k_u', help='Anisotropy constant in units of Jm^-3',
                    type=float)

parser.add_argument('--B', help='External magnetic field perpendicular to the'
                    ' square plane (z direction), in Tesla',
                    type=float)

# parser.add_argument('sk_initial_radius',
#                     help='Radius of the initial skyrmion to be relaxed ',
#                     type=float)

# parser.add_argument('--relax_time', help='Relaxation time after the SPPC '
#                     'application, in ns ',
#                     type=float)

# parser.add_argument('initial_state',
#                     help='Path to initial state .npy file')

parser.add_argument('sim_name',
                    help='Simulation name')

parser.add_argument('--pin_borders',
                    help='Pin the magnetisation vectors at the box boundaries',
                    action='store_true')

parser.add_argument('--PBC_2D',
                    help='Two dimensional boundary condition',
                    action='store_true')

initial_state.add_argument('--initial_state_skyrmion_down',
                           help='This option puts a skyrmionic texture'
                           ' in the centre of the'
                           ' nanotrack, as initial m configuration. The'
                           ' other spins are in the (0, 0, 1) direction',
                           type=float,
                           metavar=('SK_INITIAL_RADIUS')
                           # action='store_true'
                           )

initial_state.add_argument('--initial_state_skyrmion_up',
                           help='This option puts a skyrmionic texture'
                           ' with its core pointing in the +z direction, '
                           'in the centre of the'
                           ' nanotrack, as initial m configuration. The'
                           ' other spins are in the (0, 0, 1) direction',
                           type=float,
                           metavar=('SK_INITIAL_RADIUS')
                           # action='store_true'
                           )

initial_state.add_argument('--initial_state_ferromagnetic_up',
                           help='This option sets the initial '
                           'm configuration as a ferromagnetic state'
                           ' in the (0, 0, 1) direction',
                           action='store_true'
                           )

initial_state.add_argument('--initial_state_ferromagnetic_down',
                           help='This option sets the initial '
                           'm configuration as a ferromagnetic state'
                           ' in the (0, 0, -1) direction',
                           action='store_true'
                           )

initial_state.add_argument('--initial_state_irregular',
                           help='This option sets the initial '
                           'm configuration as an irregular state'
                           ' (TESTING)',
                           action='store_true')

parser.add_argument('--preview', help='Set to *yes* if a plot with m '
                    'being updated is shown instead of saving npy '
                    'and vtk files',
                    )

parser.add_argument('--alpha', help='Damping constant value',
                    type=float)

parser.add_argument('--save_files', help='Save vtk and npy files every x'
                    ' nanoseconds',
                    type=float, default=False)

# Parser arguments
args = parser.parse_args()

import dolfin as df
import numpy as np
import finmag
from finmag import Simulation as Sim
from finmag.energies import Exchange, DMI, Zeeman, Demag, UniaxialAnisotropy
import textwrap
import subprocess
import mshr

# 2D Box specification
# The center of the box will be at (0, 0, 0)
mesh = df.RectangleMesh(-args.box_length * 0.5,
                        -args.box_width * 0.5,
                        args.box_length * 0.5,
                        args.box_width * 0.5,
                        int(args.box_length / args.fd_max),
                        int(args.box_width / args.fd_max),
                        )
# df.plot(mesh, interactive=True)

print finmag.util.meshes.mesh_quality(mesh)
print finmag.util.meshes.mesh_info(mesh)

# Generate simulation object with or without PBCs
if not args.PBC_2D:
    sim = Sim(mesh, args.Ms, unit_length=1e-9,
              name=args.sim_name)
else:
    sim = Sim(mesh, args.Ms, unit_length=1e-9,
              pbc='2d', name=args.sim_name)

# Add energies
sim.add(Exchange(args.A))

if args.D != 0:
    sim.add(DMI(args.D * 1e-3, dmi_type='interfacial'))

# Zeeman field in the z direction (in Tesla)
# if it is not zero
if args.B != 0:
    sim.add(Zeeman((0, 0, args.B / finmag.util.consts.mu0)))

if args.k_u:
    sim.add(UniaxialAnisotropy(args.k_u,
            (0, 0, 1), name='Ku'))

# No Demag in 2D
# sim.add(Demag())

# sim.llg.presession = False

if args.alpha:
    sim.alpha = args.alpha

# Pin the magnetisation vectors at the borders if specified
if args.pin_borders:

    def borders_pinning(pos):
        x, y = pos[0], pos[1]

        if (x < (-args.box_length * 0.5 + args.fd_max)
                or (x > args.box_length * 0.5 - args.fd_max)):
            return True

        elif (y < (-args.box_width * 0.5 + args.fd_max)
                or (y > args.box_width * 0.5 - args.fd_max)):
            return True

        else:
            return False

    # Pass the function to the 'pins' property of the simulation
    sim.pins = borders_pinning


# Generate the skyrmion in the middle of the nanotrack
def generate_skyrmion_down(pos, sign):
    """
    Sign will affect the chirality of the skyrmion
    """
    # We will generate a skyrmion in the middle of the stripe
    # (the origin is there) with the core pointing down
    x, y = pos[0], pos[1]

    if np.sqrt(x ** 2 + y ** 2) <= args.initial_state_skyrmion_down:
        # Polar coordinates:
        r = (x ** 2 + y ** 2) ** 0.5
        phi = np.arctan2(y, x)
        # This determines the profile we want for the
        # skyrmion
        # Single twisting: k = pi / R
        k = np.pi / (args.initial_state_skyrmion_down)

        # We define here a 'hedgehog' skyrmion pointing down
        return (sign * np.sin(k * r) * np.cos(phi),
                sign * np.sin(k * r) * np.sin(phi),
                -np.cos(k * r))
    else:
        return (0, 0, 1)


def generate_skyrmion_up(pos, sign):
    # We will generate a skyrmion in the middle of the stripe
    # (the origin is there) with the core pointing down
    x, y = pos[0], pos[1]

    if np.sqrt(x ** 2 + y ** 2) <= args.initial_state_skyrmion_up:
        # Polar coordinates:
        r = (x ** 2 + y ** 2) ** 0.5
        phi = np.arctan2(y, x)
        # This determines the profile we want for the
        # skyrmion
        # Single twisting: k = pi / R
        k = np.pi / (args.initial_state_skyrmion_up)

        # We define here a 'hedgehog' skyrmion pointing down
        return (sign * np.sin(k * r) * np.cos(phi),
                sign * np.sin(k * r) * np.sin(phi),
                np.cos(k * r))
    else:
        return (0, 0, -1)


def irregular_state(pos):
    # We will generate a skyrmion in the middle of the stripe
    # (the origin is there) with the core pointing down
    x, y = pos[0], pos[1]

    if x > 0:
        return (0, 0, 1)
    else:
        return (0, 0, -1)

# Load magnetisation profile

# Change the skyrmion initial configuration according to the
# chirality of the system (give by the DMI constant)
if args.initial_state_skyrmion_down:
    if args.D > 0:
        sim.set_m(lambda x: generate_skyrmion_down(x, -1))
    else:
        sim.set_m(lambda x: generate_skyrmion_down(x, 1))

elif args.initial_state_skyrmion_up:
    if args.D > 0:
        sim.set_m(lambda x: generate_skyrmion_up(x, 1))
    else:
        sim.set_m(lambda x: generate_skyrmion_up(x, -1))

elif args.initial_state_ferromagnetic_up:
    sim.set_m((0, 0, 1))
elif args.initial_state_ferromagnetic_down:
    sim.set_m((0, 0, -1))
elif args.initial_state_irregular:
    sim.set_m(irregular_state)
else:
    raise Exception('Set one option for the initial state')

# DEBUG INFORMATION ##############################
print '\n\n'
print 'Running a {} nm x {} nm stripe'.format(args.box_length,
                                              args.box_width,
                                              )
print 'DMI constant', args.D, '* 1e-3  J m**-2'
# print 'Anisotropy constant', args.k_u, 'x 10 ** 4  Jm**-3'
print '\n\n'
# ################################################


# Save states if specified
if args.save_files:
    sim.schedule('save_vtk', every=args.save_files * 1e-9,
                 filename='{}_.pvd'.format(args.sim_name),
                 overwrite=True)
    sim.schedule('save_field', 'm', every=args.save_files * 1e-9,
                 filename='{}_.npy'.format(args.sim_name),
                 overwrite=True)
    sim.overwrite_pvd_files = True

if args.preview:
    # Save first state
    sim.save_vtk('{}_initial.pvd'.format(args.sim_name), overwrite=True)

    # Print a message with the avergae m and an updated plot
    # with the magnetisation profile
    times = np.linspace(0, 1 * 1e-9, 10000)
    for t in times:
        sim.run_until(t)
        df.plot(sim.m_field.f, interactive=False)
        print '##############################################'
        print 'Energy:  ', sim.compute_energy()
        # print sim.llg.M_average
        # print 'mz(0) = {}'.format(sim.llg._m_field.f(0, 0, 0)[2])
        print '##############################################'

else:
    # WE only need the LAST state (relaxed)
    #
    # We will save these options in case we need the transition
    # in the future:::::::::::::
    #

    # Save the initial states
    sim.save_vtk('{}_initial.pvd'.format(args.sim_name), overwrite=True)
    np.save('{}_initial.npy'.format(args.sim_name), sim.m)

    # Relax the system
    sim.relax()

    # Save the last relaxed state
    sim.save_vtk('{}_final.pvd'.format(args.sim_name), overwrite=True)
    np.save('{}_final.npy'.format(args.sim_name), sim.m)

    # Save magnetic data from the simulations last step
    sim.save_ndt()

    # Save files
    #
    npy_dir = 'npys/'
    vtk_dir = 'vtks/'
    log_dir = 'logs/'
    ndt_dir = 'ndts/'

    def mkrootdir(s):
        if not os.path.exists(s):
            os.makedirs(s)

    mkrootdir(npy_dir)
    mkrootdir(vtk_dir)
    mkrootdir(log_dir)
    mkrootdir(ndt_dir)

    files = os.listdir('.')
    for f in files:
        if ((f.endswith('vtu') or f.endswith('pvd'))
                and f.startswith(args.sim_name)):
            if os.path.exists(vtk_dir + f):
                os.remove(vtk_dir + f)

            shutil.move(f, vtk_dir)

        elif (f.endswith('npy') and f.startswith(args.sim_name)):
            if os.path.exists(npy_dir + f):
                os.remove(npy_dir + f)

            shutil.move(f, npy_dir)

        elif (f.endswith('log') and f.startswith(args.sim_name)):
            if os.path.exists(log_dir + f):
                os.remove(log_dir + f)

            shutil.move(f, log_dir)

        elif (f.endswith('ndt') and f.startswith(args.sim_name)):
            if os.path.exists(ndt_dir + f):
                os.remove(ndt_dir + f)

            shutil.move(f, ndt_dir)
