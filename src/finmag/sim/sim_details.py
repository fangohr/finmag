import finmag
import dolfin as df
import numpy as np

from finmag.util.consts import exchange_length, bloch_parameter, \
        helical_period
from finmag.util.meshes import mesh_info as mesh_information


def _get_length_scales(sim):
    lengths = {}
    if (sim.has_interaction('Exchange')):
        A = sim.get_interaction('Exchange').A_av
        Ms = sim.Ms
        l_ex = exchange_length(A, Ms)
        lengths['Exchange length'] = l_ex

        if (sim.has_interaction('Anisotropy')):
            K1 = sim.get_interaction('Anisotropy').K1.vector().array().mean()
            l_bloch = bloch_parameter(A, K1)
            lengths['Bloch parameter'] = l_bloch

        if (sim.has_interaction('DMI')):
            D = sim.get_interaction('DMI').D_av
            l_h_period = helical_period(A,D)
            lengths['Helical period'] = l_h_period

    return lengths


def length_scales(sim):
    """
    Returns a string all the relevant lengths scales (Exchange
    length, Bloch parameters and Helical period) of the sim object
    and finds the of these minimum length scales.

    First checks if the sim object has an Exchange interaction
    and issues a warning if no Exchange interaction is present.

    If the Exchange interaction is present in the sim object,
    the Exchange length is calculated as well as Bloch parameter
    (Anisotropy interaction required) and the Helical period
    (DMI interaction required).

    """
    lengths = _get_length_scales(sim)

    info_string = ""

    def added_info(name,length):
        return "The {} = {:.2f} nm.\n".format(name, length * 1e9)

    if not (sim.has_interaction('Exchange')):
        info_string += "Warning: Simulation object has no exchange. Cannot compute length scales.\n"
    else:
        for key,value in lengths.items():
            info_string += added_info(key,value)

    return info_string

def mesh_info(sim):
    """
    Return a string containing some basic information about the
    mesh (such as the number of cells, interior/surface triangles,
    vertices, etc.).

    Also print a distribution of edge lengths present in the mesh
    and how they compare to the exchange length, the Bloch
    parameter and the Helical period (if these can be computed, which
    requires an exchange interaction (plus anisotropy for the Bloch
    parameter and DMI value for the Helical period)); note that for
    spatially varying material parameters the average values are used).
    This information is relevant to estimate whether the mesh
    discretisation is too coarse and might result in numerical
    artefacts (see W. Rave, K. Fabian, A. Hubert, "Magnetic states
    ofsmall cubic particles with uniaxial anisotropy", J. Magn.
    Magn. Mater. 190 (1998), 332-348).

    """
    info_string = "{}\n".format(mesh_information(sim.mesh))

    edgelengths = [e.length() * sim.unit_length for e in df.edges(sim.mesh)]
    lengths = _get_length_scales(sim)

    def added_info(name,L):
        (a, b), _ = np.histogram(edgelengths, bins=[0, L, np.infty])
        if b == 0.0:
            msg = "All edges are shorter"
            msg2 = ""
        else:
            msg = "Warning: {:.2f}% of edges are longer".format(100.0 * b / (a + b))
            msg2 = " (this may lead to discretisation artefacts)"
        info = "{} than the {} = {:.2f} nm{}.\n".format(msg, name, L * 1e9, msg2)
        return info

    if not (sim.has_interaction('Exchange')):
        info_string += "Warning: Simulation object has no exchange. Cannot compute exchange length(s).\n"
    else:
        for key,value in lengths.items():
            info_string += added_info(key,value)

        info_string += "\nThe minimum length is the {} = {:.2f}nm".format(min(lengths,key=lengths.get),min(lengths.values())*1e9)

    return info_string
