import numpy as np
import matplotlib.colors as mcolors
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from math import pi, cos, sin

# The following values are derived from a circular path in the CIELab [1]
# color space. The path is defined as follows:
#
#   t  -->  center + radius * [0, cos(t), sin(t)]
#
# (where t ranges from 0 to 2*pi).
#
# This path was sampled at 256 equidistant values of t and the resulting
# CIELab coordinats were converted back into RGB coordinates, resulting in
# the values below.
#
# The center and radius need to be chosen so that the converted coordinates
# all represent valid RGB values.
#
#
# [1] http://dba.med.sc.edu/price/irf/Adobe_tg/models/cielab.html

def rgb2lab(pt):
    return convert_color(sRGBColor(pt[0], pt[1], pt[2]), LabColor).get_value_tuple()

def lab2rgb(pt):
    return convert_color(LabColor(pt[0], pt[1], pt[2]), sRGBColor).get_value_tuple()

# L=50 (halfway between black and white)
center1 = np.array([50, 25, 5])
radius1 = 52

# L=60 (somewhat lighter than the previous one)
center2 = np.array([60, 15, 13])
radius2 = 51

# L=65 (even lighter)
center3 = np.array([65, 5, 18])
radius3 = 49

# L=65, but path is centered around the L=0 axis
center4 = np.array([65, 0, 0])
radius4 = 36


def circular_colormap(center, radius, offset=0):
    """
    Define a perceptually linear, circular colormap defined through a
    circular path in the CIELab [1] color space. The path is defined
    by the arguments `center` and `radius` via the mapping:

       t  -->  center + radius * [0, cos(t + offset), sin(t + offset)]

    where t ranges from 0 to 2*pi. The argument `offset` can be used to
    shift the colors along the colorscale.

    It is up to the user to choose values of `center` and `radius` so
    that the entire path lies within the range representing valid RGB
    values.

    [1] http://dba.med.sc.edu/price/irf/Adobe_tg/models/cielab.html

    """
    center = np.asarray(center)

    tvals = np.linspace(0, 2 * pi, 256, endpoint=False)
    path_vals = [center + radius * np.array([0, cos(t+offset), sin(t+offset)]) for t in tvals]

    cmap_vals = np.array([lab2rgb(pt) for pt in path_vals])
    cmap = mcolors.ListedColormap(cmap_vals)
    return cmap


def linear_colormap(rgb1, rgb2):
    """
    Define a perceptually linear colormap defined through a line in the
    CIELab [1] color space.

    [1] http://dba.med.sc.edu/price/irf/Adobe_tg/models/cielab.html

    """
    pt1 = np.array(rgb2lab(rgb1))
    pt2 = np.array(rgb2lab(rgb2))

    tvals = np.linspace(0, 1, 256)
    path_vals = [(1-t) * pt1 + t * pt2 for t in tvals]
    cmap_vals = np.array([lab2rgb(pt) for pt in path_vals])
    cmap = mcolors.ListedColormap(cmap_vals)
    return cmap


# Define a few colormaps with increasingly lighter colors
circular1 = circular_colormap(center=[50, 25, 5], radius=52)
circular2 = circular_colormap(center=[60, 15, 13], radius=51)
circular3 = circular_colormap(center=[65, 5, 18], radius=49)
