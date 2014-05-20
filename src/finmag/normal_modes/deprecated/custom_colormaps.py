import numpy as np
import matplotlib.colors as mcolors
import husl
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


def circular_colormap(center, radius, normal_vector=[1, 0, 0], offset=0):
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
    n = np.asarray(normal_vector)
    n = n / np.linalg.norm(n)

    arbitrary_direction = np.array([0.28, 0.33, 0.71])  # arbitrary vector to make sure that e1 below is unlikely to be zero

    # Define two orthogonal vectors e1, e2 lying in the plane orthogonal to `normal_vector`.
    e1 = np.cross(n, arbitrary_direction)
    e2 = np.cross(n, e1)

    tvals = np.linspace(0, 2 * pi, 256, endpoint=False)
    path_vals = [center + radius * cos(t+offset) * e1 + radius * sin(t+offset) * e2 for t in tvals]

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


def husl_colormap(saturation, lightness):
    """
    Generate a colormap of linearly varying hue while keeping saturation and lightness fixed.
    See here for the HUSL color space: http://www.boronine.com/husl/

    """
    hvals = np.linspace(0, 360, 256, endpoint=False)
    cmap_vals = np.array([husl.husl_to_rgb(h, saturation, lightness) for h in hvals])
    cmap = mcolors.ListedColormap(cmap_vals)
    return cmap


# Define a few colormaps with increasingly lighter colors
circular1 = circular_colormap(center=[50, 25, 5], radius=52)
circular2 = circular_colormap(center=[60, 15, 13], radius=51)
circular3 = circular_colormap(center=[65, 5, 18], radius=49)
circular4 = circular_colormap(center=[60, 6, 24], radius=51, normal_vector=[3, 1, -1])

husl_99_75 = husl_colormap(saturation=99, lightness=75)
husl_99_70 = husl_colormap(saturation=99, lightness=70)
husl_99_65 = husl_colormap(saturation=99, lightness=65)
