import finmag
from finmag.util import helpers

from copy import copy
import logging
import math
import numpy as np
from numpy.linalg import norm

log = logging.getLogger("finmag")


def initialise_helix_2D(sim, period, axis=np.array([1, 0])):
    """
    Initialise the magnetisation to a pattern resembling a helix along a given
    direction. By default, helices along the first dimension will be created.
    The resulting magnetisation should be relaxed to obtain the most natural
    helical state possible.

    *Assumptions*: It is assumed that the mesh is a 2D mesh. If it is not, the
                   skyrmion will be projected in higher dimensions.

    *Arguments*:

    period
        Float representing the period of the helix in mesh co-ordinates.

    axis
        Numpy array containing direction of helix.

    This function returns nothing.
    """

    # Check for sane values.
    if np.linalg.norm(axis) == 0:
        raise ValueError("Axis ({}) norm is zero.".format(axis))

    if period <= 0:
        raise ValueError("Helix period ({}) cannot be zero or less."
                         .format(period))

    # Normalise the axis.
    axis = axis / np.linalg.norm(axis)

    # Create rotation matrix to convert mesh co-ordinates to co-ordinates of
    # the axis and it's right-handed perpendicular. In this new system, the
    # second dimension corresponds to the axis direction.
    cos_theta = axis[0]
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)

    R = np.matrix(([cos_theta, sin_theta],
                   [-sin_theta, cos_theta]))

    # Build the function.
    def m_helical(pos):

        # Convert to rotated co-ordinate system.
        pos_rot_matrix = R * np.transpose(np.matrix(pos))
        pos_rot = np.array((pos_rot_matrix.item(0), pos_rot_matrix.item(1)))

        # Find magnetisation components.
        mx_r = np.cos((1 - (pos_rot[1] / period)) * 2 * np.pi)
        my_r = 0
        mz = np.sin((1 - (pos_rot[1] / period)) * 2 * np.pi)

        # Rotate magnetisation back into mesh co-ordinates.
        m = R * np.matrix(([mx_r], [my_r]))
        mx = m.item(0)
        my = m.item(1)

        # Normalise and return.
        out = np.array([mx, my, mz], dtype="float64")
        return out / np.linalg.norm(out)

    # Use the above function to initialise the magnetisation.
    sim.set_m(m_helical)


def initialise_skyrmions(sim, skyrmionRadius, centres=np.array([[0, 0]])):
    """
    Initialise the magnetisation to a pattern resembling skyrmions with defined
    centres. By default, a single skyrmion at (0, 0) will be created. The
    resulting magnetisation should be relaxed to obtain the most natural
    skyrmion possible. If skyrmions are found to overlap, a ValueError will be
    raised. If centres is empty, the ferromagnetic state will be initialised.

    *Assumptions*: It is assumed that the mesh is a 2D mesh. If it is not, the
                   skyrmion will be projected in higher dimensions.

    *Arguments*:

    centres
        Numpy array containing co-ordinates stored by numpy arrays.

    skyrmionRadius
        The radius of the skyrmion in mesh co-ordinates (single value for all
        skyrmions).

    This function returns nothing.
    """

    numCentres = len(centres)

    # Initialise ferromagnetic state if there are no centres.
    if numCentres == 0:
        def m_ferromagnetic(pos):
            return np.array([0, 0, 1], dtype="float64")
        sim.set_m(m_ferromagnetic)
        return

    # Determine whether there is skyrmion overlap.
    if numCentres > 1:
        for zI in xrange(numCentres):
            for zJ in xrange(zI + 1, numCentres):
                if norm(centres[zI] - centres[zJ]) < 2 * skyrmionRadius:
                    raise ValueError("Skyrmions at centres {} and {} overlap."
                                     .format(centres[zI], centres[zJ]))

    # Build the function
    def m_skyrmions(pos):

        # For each skyrmion, check if the position vector exists within it.
        # Will pass until one is obtained.
        for zI in xrange(numCentres):

            loc = copy(pos)  # Assignment means the original will change if "="
                             # operator is used.

            # Convert the position vector into relative vector with respect to
            # this centre.
            loc = loc - centres[zI]

            # Find the radius component in cylindrical form.
            r = norm(loc)

            # Check to see if this vector is within this circle. If it isn't,
            # check the next centre by skipping the rest of this iteration.
            if r > skyrmionRadius:
                continue

            # Convert position into cylindrical form, using r defined
            # previously, and "t" as the argument.
            if abs(loc[0]) < 1e-6:
                if abs(loc[1]) < 1e-6:
                    t = 0
                elif loc[1] > 0:
                    t = np.pi / 2.
                else:
                    t = -np.pi / 2.
            else:
                t = np.arctan2(loc[1], loc[0])

            # Define vector components inside the skyrmion:
            mz = -np.cos(np.pi * r / skyrmionRadius)
            mt = np.sin(np.pi * r / skyrmionRadius)

            # Convert to cartesian form and normalize.
            mx = -np.sin(t) * mt
            my = np.cos(t) * mt
            out = np.array([mx, my, mz], dtype="float64")
            return out / norm(out)

        # If control reaches here, it means that the vector is not in any
        # circle. As such, return the corresponding ferromagnetic-state-like
        # vector.
        return np.array([0., 0., 1.], dtype="float64")

    # Use the above function to initialise the magnetisation.
    sim.set_m(m_skyrmions)


def initialise_skyrmion_hexlattice_2D(sim, meshX, meshY, tileScaleX=1,
                                      skyrmionRadiusScale=0.2):
    """
    Initialise the magnetisation to a pattern resembling a hexagonal
    lattice of 2D skyrmions which should be relaxed to obtain the most
    natural lattice possible. The most stable of lattices will exist on
    meshes whose horizontal dimension is equal to the vertical tile
    dimension / sqrt(3); if the mesh does not (approximately) conform to
    these dimensions, a warning will be raised (because this will result in
    elliptical skyrmions being created).

    *Assumptions*: It is assumed that the mesh is a rectangular 2D mesh. If
                   it is not, the 2D hexagonal lattice will be projected in
                   higher dimensions.

    *Arguments*:

    meshX
        The length of the rectangular mesh in the first dimension (the 'X'
        direction).

    meshY
        As above, but for the second dimension (the 'Y' direction).

    tileScaleX=1
        The fraction of the size of the tile in the first dimension (the
        'X' direction) to the size of the mesh. For example, if this
        quantity is equal to 0.5, there will be two tiles in the first
        dimension.

    skyrmionRadiusScale=0.3
        The radius of the skyrmion defined as a fraction of the tile in the
        first dimension ('X' direction).

    This function returns nothing.
    """

    # Ensure the mesh is of the dimensions stated above, and raise a
    # warning if it isn't.
    ratioX = abs(meshX / float(meshY) - np.sqrt(3))
    ratioY = abs(meshY / float(meshX) - np.sqrt(3))
    if ratioX > 0.05 and ratioY > 0.05:
        log.warning("Mesh dimensions do not accurately support hexagonal" +
                    " lattice formation! (One should be a factor of sqrt" +
                    "(3) greater than the other.)")

    # Calculate lengths of tiles and the skyrmion radius in mesh
    # co-ordinates.
    tileLengthX = meshX * tileScaleX
    tileLengthY = meshY * tileScaleX
    skyrmionRadius = tileLengthX * skyrmionRadiusScale

    # Build the function.
    def m_skyrmion_hexlattice(pos):
        """
        Function that takes a position vector of a point in a vector field
        and returns a vector such that the vector field forms a hexagonal
        lattice of skyrmions of the field is toroidal. This only works for
        rectangular meshes whos horizontal tile dimension is equal to the
        vertical tile dimension / sqrt(3)."""

        # Convert position into tiled co-ordinates.
        cx = pos[0] % tileLengthX - tileLengthX / 2.
        cy = pos[1] % tileLengthY - tileLengthY / 2.

        # ==== Define first skyrmion quart ====#
        # Define temporary cartesian co-ordinates (tcx, tcy) that can be
        # used to define a polar co-ordinate system.
        tcx = cx - tileLengthX / 2.
        tcy = cy

        r = pow(pow(tcx, 2) + pow(tcy, 2), 0.5)
        t = np.arctan2(tcy, tcx)

        tcx_flip = cx + tileLengthX / 2.
        r_flip = pow(pow(tcx_flip, 2) + pow(tcy, 2), 0.5)
        t_flip = np.arctan2(tcy, tcx_flip)

        # Populate vector field:
        mz = -1 + 2 * r / skyrmionRadius
        mz_flip = -1 + 2 * r_flip / skyrmionRadius

        # Replicate to other half-plane of the vector field and convert to
        # cartesian form.
        if(mz <= 1):
            mt = np.sin(np.pi * r / skyrmionRadius)
            mx_1 = -np.sin(t) * mt
            my_1 = np.cos(t) * mt
            mz_1 = mz

        elif(mz_flip < 1):
            mt = -np.sin(np.pi * r_flip / skyrmionRadius)
            mz_1 = mz_flip
            mx_1 = np.sin(t_flip) * mt
            my_1 = -np.cos(t_flip) * mt

        elif(mz > 1):
            mx_1 = 0
            my_1 = 0
            mz_1 = 1

        # ==== Define second skyrmion quart ====#
        # Define temporary cartesian co-ordinates (tcx, tcy) that can be
        # used to define polar co-ordinate system.
        tcx = cx
        tcy = cy - tileLengthY / 2.

        r = pow(pow(tcx, 2) + pow(tcy, 2), 0.5)
        t = np.arctan2(tcy, tcx)

        tcy_flip = cy + tileLengthY / 2.
        r_flip = pow(pow(tcx, 2) + pow(tcy_flip, 2), 0.5)
        t_flip = np.arctan2(tcy_flip, tcx)

        # Populate vector field:
        mz = -1 + 2 * r / skyrmionRadius
        mz_flip = -1 + 2 * r_flip / skyrmionRadius

        # Replicate to other half-plane of the vector field and convert to
        # cartesian form.
        if(mz <= 1):
            mt = np.sin(np.pi * r / skyrmionRadius)
            mx_2 = -np.sin(t) * mt
            my_2 = np.cos(t) * mt
            mz_2 = mz

        elif(mz_flip < 1):
            mt = -np.sin(np.pi * r_flip / skyrmionRadius)
            mz_2 = mz_flip
            mx_2 = np.sin(t_flip) * mt
            my_2 = -np.cos(t_flip) * mt

        elif(mz > 1):
            mx_2 = 0
            my_2 = 0
            mz_2 = 1

        #==== Combine and normalize. ====#
        mx = mx_1 + mx_2
        my = my_1 + my_2
        mz = mz_1 + mz_2 - 1

        out = np.array([mx, my, mz], dtype="float64")

        return out / norm(out)

    # Use the above function to initialise the magnetisation.
    sim.set_m(m_skyrmion_hexlattice)


def vortex_simple(r, center, right_handed=True, polarity=+1):
    """
    Returns a function f: (x,y,z) -> m representing a vortex magnetisation
    pattern where the vortex lies in the x/y-plane (i.e. the magnetisation is
    constant along the z-direction), the vortex core is centered around the
    point `center` and the vortex core has radius `r`. More precisely, m_z=1 at
    the vortex core center and m_z falls off in a radially symmetric way until
    m_z=0 at a distance `r` from the center. If `right_handed` is True then the
    vortex curls counterclockwise around the z-axis, otherwise clockwise. It
    should be noted that the returned function `f` only represents an
    approximation to a true vortex state, so this can be used to initialise the
    magnetisation in a simulation which is then relaxed.

    Note that both `r` and the coordinates of `center` should be given in mesh
    coordinates, not in metres.

    """
    def f(pt):
        x, y, z = pt
        xc = x - center[0]
        yc = y - center[1]
        phi = math.atan2(yc, xc)
        rho = math.sqrt(xc ** 2 + yc ** 2)

        # To start with, create a right-handed vortex with polarity 1.
        if rho < r:
            theta = 2 * math.atan(rho / r)
            mz = math.cos(theta)
            mx = -math.sin(theta) * math.sin(phi)
            my = math.sin(theta) * math.cos(phi)
        else:
            mz = 0
            mx = -math.sin(phi)
            my = math.cos(phi)

        # If we actually want a different polarity, flip the z-coordinates
        if polarity < 0:
            mz = -mz

        # Adapt the chirality accordingly
        if ((polarity > 0) and (not right_handed)) or\
           ((polarity < 0) and right_handed):
            mx = -mx
            my = -my

        return (mx, my, mz)

    return f


def vortex_feldtkeller(beta, center, right_handed=True, polarity=+1):
    """
    Returns a function f: (x,y,z) -> m representing a vortex
    magnetisation pattern where the vortex lies in the x/y-plane (i.e.
    the magnetisation is constant along the z-direction), the vortex
    core is centered around the point `center` and the m_z component
    falls off exponentially as given by the following formula, which
    is a result by Feldtkeller and Thomas [1].:

        m_z = exp(-2*r^2/beta^2).

    Here `r` is the distance from the vortex core centre and `beta` is
    a parameter, whose value is taken from the first argument to this
    function.

    [1] E. Feldtkeller and H. Thomas, "Struktur und Energie von
        Blochlinien in Duennen Ferromagnetischen Schichten", Phys.
        Kondens. Materie 8, 8 (1965).

    """
    beta_sq = beta ** 2

    def f(pt):
        x, y, z = pt
        # To start with, create a right-handed vortex with polarity 1.
        xc = x - center[0]
        yc = y - center[1]
        phi = math.atan2(yc, xc)
        r_sq = xc ** 2 + yc ** 2
        mz = math.exp(-2.0 * r_sq / beta_sq)
        mx = -math.sqrt(1 - mz * mz) * math.sin(phi)
        my = math.sqrt(1 - mz * mz) * math.cos(phi)

        # If we actually want a different polarity, flip the z-coordinates
        if polarity < 0:
            mz = -mz

        # Adapt the chirality accordingly
        if ((polarity > 0) and (not right_handed)) or\
           ((polarity < 0) and right_handed):
            mx = -mx
            my = -my

        return (mx, my, mz)

    return f


def initialise_vortex(sim, type, center=None, **kwargs):
    """
    Initialise the magnetisation to a pattern that resembles a vortex state.
    This can be used as an initial guess for the magnetisation, which should
    then be relaxed to actually obtain the true vortex pattern (in case it is
    energetically stable).

    If `center` is None, the vortex core centre is placed at the sample centre
    (which is the point where each coordinate lies exactly in the middle
    between the minimum and maximum coordinate for each component). The vortex
    lies in the x/y-plane (i.e. the magnetisation is constant in z-direction).
    The magnetisation pattern is such that m_z=1 in the vortex core centre, and
    it falls off in a radially symmetric way.

    The exact vortex profile depends on the argument `type`. Currently the
    following types are supported:

       'simple':

           m_z falls off in a radially symmetric way until m_z=0 at
           distance `r` from the centre.

       'feldtkeller':

           m_z follows the profile m_z = exp(-2*r^2/beta^2), where `beta`
           is a user-specified parameter.

    All provided keyword arguments are passed on to functions which implement
    the vortex profiles. See their documentation for details and other allowed
    arguments.

    """
    coords = np.array(sim.mesh.coordinates())
    if center is None:
        center = 0.5 * (coords.min(axis=0) + coords.max(axis=0))

    vortex_funcs = {'simple': vortex_simple,
                    'feldtkeller': vortex_feldtkeller}

    kwargs['center'] = center

    try:
        fun_m_init = vortex_funcs[type](**kwargs)
        log.debug("Initialising vortex of type '{}' with arguments: {}".
                  format(type, kwargs))
    except KeyError:
        raise ValueError("Vortex type must be one of {}. Got: {}".
                         format(vortex_funcs.keys(), type))

    sim.set_m(fun_m_init)
